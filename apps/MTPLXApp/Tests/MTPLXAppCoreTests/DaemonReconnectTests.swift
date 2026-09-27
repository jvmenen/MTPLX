import Darwin
import Foundation
import XCTest

@testable import MTPLXAppCore

// Issue #528: the header badge stuck at "Degraded — MTPLX is already
// running." over a daemon that answered /health and served requests the
// whole time, and only quitting the app cleared it. A second start request
// against the daemon the app already ran (the window's launch task running
// again after a window reopened, Start pressed while a model loaded, the
// benchmark or Hermes asking for a ready daemon) made the supervisor throw
// `alreadyRunning`; the store published that as Degraded, cancelled the
// watchdog, and waited for a launch id that never existed.
//
// These tests drive the real store and supervisor against a scripted
// `mtplx` (a small Python HTTP server, no model), the same way the app does.

/// A scripted `mtplx serve` for store lifecycle tests. It answers the
/// endpoints the store reads, reports its own pid and launch id on /health,
/// appends every spawn to `spawns.log`, and takes switches through files so a
/// test can make /health or the live-stats stream fail.
struct ReconnectFakeDaemon {
    let root: URL
    let executable: URL
    let modelDirectory: URL
    let port: Int

    struct Spawn: Equatable {
        let pid: Int
        let launchID: String?
    }

    static func make() throws -> ReconnectFakeDaemon {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-reconnect-\(UUID().uuidString)", isDirectory: true)
        let bin = root.appendingPathComponent("bin", isDirectory: true)
        let model = root.appendingPathComponent("complete-model", isDirectory: true)
        let runtime = root.appendingPathComponent("runtime", isDirectory: true)
        for directory in [bin, model, runtime] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
        for file in ["config.json", "tokenizer.json", "mtplx_runtime.json"] {
            try Data("{}".utf8).write(to: model.appendingPathComponent(file))
        }
        try Data([0]).write(to: model.appendingPathComponent("mtp.safetensors"))
        try Data([0]).write(to: model.appendingPathComponent("model.safetensors"))

        let server = bin.appendingPathComponent("fake_mtplx.py")
        try Data(serverSource(controlDirectory: root).utf8).write(to: server)
        let executable = bin.appendingPathComponent("mtplx")
        try writeExecutable(
            "#!/bin/sh\nexec python3 -u \(shellQuoted(server.path)) \"$@\"\n",
            to: executable
        )
        // What the launch path's runtime check resolves; it only asks for a
        // version and never starts this.
        try writeExecutable(
            "#!/bin/sh\necho 'mtplx 2.12.0 (2.12.0)'\n",
            to: runtime.appendingPathComponent("mtplx")
        )
        return ReconnectFakeDaemon(
            root: root,
            executable: executable,
            modelDirectory: model,
            port: try freeTCPPort()
        )
    }

    var baseURL: URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }

    func spawns() -> [Spawn] {
        let text = (try? String(contentsOf: root.appendingPathComponent("spawns.log"), encoding: .utf8)) ?? ""
        return text.split(separator: "\n").compactMap { line in
            let fields = line.split(separator: " ")
            guard fields.count == 2, let pid = Int(fields[0]) else { return nil }
            return Spawn(pid: pid, launchID: fields[1] == "-" ? nil : String(fields[1]))
        }
    }

    /// /health answers 503 while set: a daemon still loading its model, or
    /// one busy in a long commit (issue #487).
    func setHealthDown(_ down: Bool) throws {
        try setFlag("health-down", down)
    }

    /// The live-stats stream closes and refuses new connections while set:
    /// what the app sees across a sleep or a network blip.
    func setStreamDown(_ down: Bool) throws {
        try setFlag("stream-down", down)
    }

    /// A daemon the app did not launch in this session: an app-owned one
    /// from an earlier session (with a launch id) or `mtplx serve` typed in
    /// a terminal (without one).
    func launchOutsideTheApp(launchID: String?) throws -> Process {
        let process = Process()
        process.executableURL = executable
        process.arguments = ["serve", "--port", String(port), "--model", modelDirectory.path]
        var environment = ProcessInfo.processInfo.environment
        environment.removeValue(forKey: "MTPLX_APP_LAUNCH_ID")
        if let launchID {
            environment["MTPLX_APP_LAUNCH_ID"] = launchID
        }
        process.environment = environment
        try process.run()
        return process
    }

    func waitUntilHealthy(timeout: TimeInterval = 10) async throws -> HealthPayload {
        let client = MTPLXAPIClient(baseURL: baseURL)
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if let health = try? await client.health(), health.ok {
                return health
            }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        throw DaemonSupervisorError.healthTimeout
    }

    /// The settings a store under test launches with. Onboarding is done
    /// and launch-on-open is on, as in the report.
    func configuration(fanMode: MTPLXFanMode) -> MTPLXAppConfiguration {
        MTPLXAppConfiguration(
            executablePath: executable.path,
            model: modelDirectory.path,
            port: port,
            launchDaemonOnOpen: true,
            fanMode: fanMode.rawValue,
            pinFansAtMaxOnStart: fanMode == .max,
            lastLaunchTarget: LaunchTarget.chat.rawValue,
            onboardingCompletedAt: Date(),
            customModels: [
                MTPLXModelOption(
                    id: "complete-model",
                    displayName: "Complete Model",
                    shortName: "Complete",
                    detail: "Fixture model",
                    hfModelID: "Example/CompleteModel",
                    localCandidates: [modelDirectory.path]
                )
            ]
        )
    }

    var settingsStore: MTPLXSettingsStore {
        MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json"))
    }

    /// A store wired like the app's, with every side effect outside this
    /// fixture replaced: fans are recorded, never driven, and the runtime
    /// check resolves the fixture's own `mtplx` without touching the network.
    @MainActor
    func makeStore(
        configuration: MTPLXAppConfiguration,
        supervisor: DaemonSupervisor = DaemonSupervisor(),
        fans: FanCallRecorder
    ) -> MTPLXBackendStore {
        MTPLXBackendStore(
            configuration: configuration,
            settingsStore: settingsStore,
            supervisor: supervisor,
            runtimeUpdateService: MTPLXRuntimeUpdateService(
                manifestURL: URL(string: "http://127.0.0.1:1/releases/latest.json")!,
                environment: [
                    "PATH": root.appendingPathComponent("runtime").path + ":/usr/bin:/bin",
                    "HOME": root.path,
                    "MTPLX_APP_DISABLE_STANDARD_PATHS": "1",
                ]
            ),
            localFanRestorer: { await fans.restore() },
            fanModeSetter: { _, mode, _, _ in
                await fans.set(mode)
                return FanModeResponse(verified: true, currentMode: mode)
            }
        )
    }

    private func setFlag(_ name: String, _ on: Bool) throws {
        let url = root.appendingPathComponent(name)
        if on {
            try Data().write(to: url)
        } else if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    private static func writeExecutable(_ text: String, to url: URL) throws {
        try Data(text.utf8).write(to: url)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
    }

    private static func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    private static func freeTCPPort() throws -> Int {
        let socketFD = socket(AF_INET, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        defer { Darwin.close(socketFD) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = in_port_t(0).bigEndian
        address.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))
        var bindAddress = address
        let bindResult = withUnsafePointer(to: &bindAddress) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(socketFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        var bound = sockaddr_in()
        let nameResult = withUnsafeMutablePointer(to: &bound) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(socketFD, $0, &length)
            }
        }
        guard nameResult == 0 else {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .ENOTSUP)
        }
        return Int(UInt16(bigEndian: bound.sin_port))
    }

    private static func serverSource(controlDirectory: URL) -> String {
        #"""
        import json
        import os
        import signal
        import sys
        import threading
        import time
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        if "--version" in sys.argv:
            print("mtplx 2.12.0 (2.12.0)")
            sys.exit(0)

        def arg(name, default=None):
            if name in sys.argv:
                index = sys.argv.index(name)
                if index + 1 < len(sys.argv):
                    return sys.argv[index + 1]
            return default

        CONTROL = r'''__CONTROL__'''
        PORT = int(arg("--port", "0"))
        MODEL = arg("--model", "/models/test")
        FAN_MODE = arg("--fan-mode", "default")
        LAUNCH_ID = arg("--app-launch-id") or os.environ.get("MTPLX_APP_LAUNCH_ID") or None
        PID = os.getpid()
        PARENT = os.getppid()

        def control(name):
            return os.path.join(CONTROL, name)

        def flag(name):
            return os.path.exists(control(name))

        with open(control("spawns.log"), "a", encoding="utf-8") as handle:
            handle.write("%d %s\n" % (PID, LAUNCH_ID or "-"))

        def guard():
            # Never outlive the test process, and never run for long.
            deadline = time.time() + 180
            while True:
                if os.getppid() != PARENT or time.time() > deadline:
                    os._exit(0)
                time.sleep(0.5)

        threading.Thread(target=guard, daemon=True).start()
        signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
        signal.signal(signal.SIGINT, lambda *_: os._exit(0))

        HEALTH = {
            "ok": True,
            "model": "mtplx-test-model",
            "model_path": MODEL,
            "generation_mode": "mtp",
            "load_mtp": True,
            "mtp_enabled": True,
            "depth": 3,
            "profile": {"name": "sustained"},
            "context_window": 4096,
            "max_response_tokens": 1024,
            "active_requests": 0,
            "reasoning_parser": "qwen3",
            "chip": "Apple M5 Max",
            "machine_model": "Mac16,1",
            "unified_memory_bytes": 137438953472,
            "startup": {
                "launch_id": LAUNCH_ID,
                "pid": PID,
                "started_at": 1.0,
                "model_id": "mtplx-test-model",
                "warmup": {"ok": True},
            },
            "thermal": {
                "max_requested": True,
                "max_verified": True,
                "actual_ramp_verified": True,
                "fan_summary": {"ok": True},
                "verified_at": "1.0",
                "verified": {"ok": True},
            },
        }

        SETTINGS = {"depth": 3, "temperature": 0.6, "top_p": 0.95, "top_k": 20}

        SNAPSHOT = {
            "ts": 1.0,
            "model_id": "mtplx-test-model",
            "profile": {"name": "sustained"},
            "context_window": 4096,
            "active_requests": 0,
            "in_flight": [],
            "latest": {"decode_tok_s": 55.0, "session_id": "s1"},
            "recent": [],
            "rolling": {
                "window_s": 300.0, "count": 1, "min": 55.0, "max": 55.0,
                "mean": 55.0, "p50": 55.0, "p95": 55.0,
                "history": [{"t": 1.0, "tok_s": 55.0, "session_id": "s1"}],
                "live_history": [], "sticky_all_time_max": 55.0,
            },
            "lifetime": {
                "started_at_s": 1.0, "uptime_s": 2.0, "prompt_tokens_total": 10,
                "completion_tokens_total": 20, "cached_tokens_total": 5,
                "tokens_total": 30, "requests_total": 1, "cancelled_total": 0,
            },
            "sessions": {"sessions": [], "count": 0, "session_bank": {"prefixes": []}},
            "session_bank": {"prefixes": []},
            "mem": {"ok": True},
            "thermal": None,
            "thermal_when_s": 0.0,
            "settings": SETTINGS,
            "machine": {"chip": "Apple M5 Max", "machine_model": "Mac16,1", "unified_memory_bytes": 137438953472},
            "uptime_s": 2.0,
        }

        CAPABILITIES = {
            "ok": True,
            "name": "MTPLX test daemon",
            "api_version": 1,
            "endpoints": {},
            "mutable_settings": [],
            "restart_required_settings": [],
            "snapshot_interval": {
                "default_ms": 500, "min_ms": 250, "max_ms": 5000,
                "native_default_ms": 500, "performance_lock_ms": 1000,
            },
            "features": {},
            "scheduler": {},
        }

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def _json(self, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _empty(self, status):
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    if flag("health-down"):
                        return self._empty(503)
                    return self._json(HEALTH)
                if path == "/v1/mtplx/metrics/stream":
                    if flag("stream-down"):
                        return self._empty(503)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    frame = ("event: snapshot\ndata: " + json.dumps(SNAPSHOT) + "\n\n").encode("utf-8")
                    try:
                        while not flag("stream-down"):
                            self.wfile.write(frame)
                            self.wfile.flush()
                            time.sleep(0.2)
                    except OSError:
                        pass
                    return
                if path == "/v1/mtplx/app/capabilities":
                    return self._json(CAPABILITIES)
                if path == "/admin/sessions":
                    return self._json({"sessions": [], "count": 0})
                if path == "/v1/mtplx/prefill_history":
                    return self._json({"capacity": 0, "history": []})
                if path == "/v1/models":
                    return self._json({
                        "object": "list",
                        "data": [{"id": "mtplx-test-model", "object": "model", "owned_by": "mtplx"}],
                    })
                if path == "/v1/mtplx/thermal/status":
                    return self._json({
                        "ok": True,
                        "current_mode": FAN_MODE,
                        "detection": {"available": True},
                        "fan_summary": {"ok": True},
                    })
                if path == "/v1/mtplx/snapshot":
                    return self._json(SNAPSHOT)
                if path == "/v1/mtplx/settings":
                    return self._json(SETTINGS)
                return self._empty(404)

            def do_POST(self):
                path = self.path.split("?", 1)[0]
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length:
                    self.rfile.read(length)
                if path == "/v1/mtplx/settings":
                    return self._json(SETTINGS)
                return self._empty(404)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        Server(("127.0.0.1", PORT), Handler).serve_forever()
        """#.replacingOccurrences(of: "__CONTROL__", with: controlDirectory.path)
    }
}

/// Records the fan side effects a store would have caused. Nothing here
/// drives real fans.
actor FanCallRecorder {
    private(set) var restores = 0
    private(set) var modes: [String] = []

    func restore() -> Bool {
        restores += 1
        return true
    }

    func set(_ mode: String) {
        modes.append(mode)
    }
}

/// Holds the first call through it until the test opens it; later calls
/// pass straight through. Put in front of the supervisor's first /health
/// probe, it suspends a start at its first await, where a Stop can overtake
/// it.
actor FirstCallGate {
    private var used = false
    private var entered = false
    private var opened = false
    private var enteredWaiters: [CheckedContinuation<Void, Never>] = []
    private var release: CheckedContinuation<Void, Never>?

    func pass() async {
        guard !used else { return }
        used = true
        entered = true
        enteredWaiters.forEach { $0.resume() }
        enteredWaiters.removeAll()
        guard !opened else { return }
        await withCheckedContinuation { release = $0 }
    }

    func waitUntilEntered() async {
        if entered { return }
        await withCheckedContinuation { enteredWaiters.append($0) }
    }

    func open() {
        opened = true
        release?.resume()
        release = nil
    }
}

final class DaemonReconnectTests: XCTestCase {
    override func setUp() {
        super.setUp()
        L10n.activate(.english)
    }

    /// The daemon half of MTPLXApp.swift's window `.task` for a user who
    /// finished onboarding. SwiftUI runs it again whenever a new main window
    /// appears, which is one way a second start reaches a running daemon.
    @MainActor
    private func runWindowLaunchTask(_ store: MTPLXBackendStore) async {
        store.loadPersistedSettings()
        if store.configuration.launchDaemonOnOpen {
            await store.startDaemon()
        } else {
            await store.attachExistingDaemonIfOwned()
        }
    }

    @MainActor
    private func badge(_ store: MTPLXBackendStore) -> DaemonStatusBadge {
        DaemonStatusBadge(daemonState: store.daemonState, connectionState: store.connectionState)
    }

    /// The Logs window is what a user pastes into a report; it must say why
    /// a start request did not launch anything.
    @MainActor
    private func logsMention(_ store: MTPLXBackendStore, _ needle: String) async -> Bool {
        for _ in 0..<50 {
            await store.refreshLogs()
            if store.logs.contains(where: { $0.message.contains(needle) }) { return true }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        return false
    }

    // MARK: A second start against the app's own daemon

    @MainActor
    func testSecondStartAgainstItsOwnDaemonStaysRunning() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let fans = FanCallRecorder()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .max), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)
        try await pollUntil("live stats open") { store.connectionState == .open }
        let launched = try XCTUnwrap(store.health?.startup)
        let restoresBefore = await fans.restores

        // The window's launch task runs again, or Start is pressed.
        await store.startDaemon()

        XCTAssertEqual(
            store.daemonState, .running,
            "a second start against the app's own healthy daemon must not degrade"
        )
        XCTAssertEqual(store.startupPhase, .ready)
        XCTAssertEqual(store.health?.startup?.launchId, launched.launchId)
        XCTAssertEqual(store.health?.startup?.pid, launched.pid)
        XCTAssertEqual(daemon.spawns().count, 1, "a second start must not launch a second daemon")
        let restoresAfter = await fans.restores
        XCTAssertEqual(restoresAfter, restoresBefore, "the running daemon's max fans must not be reset")
        try await pollUntil("live stats still open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
        XCTAssertEqual(badge(store).tone, .healthy)
        let logged = await logsMention(store, "reconnecting to it instead of launching")
        XCTAssertTrue(logged, "the Logs window records why no second launch happened")
    }

    // MARK: A second start against an adopted daemon

    @MainActor
    func testSecondLaunchTaskAgainstAnAdoptedDaemonStaysRunning() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let prior = try daemon.launchOutsideTheApp(launchID: "prior-session-\(UUID().uuidString)")
        addTeardownBlock { if prior.isRunning { prior.terminate() } }
        let priorHealth = try await daemon.waitUntilHealthy()
        let fans = FanCallRecorder()
        // The settings file on disk is what the window task loads (report 2
        // edited it while the app was closed).
        try daemon.settingsStore.save(daemon.configuration(fanMode: .default))
        let store = daemon.makeStore(configuration: MTPLXAppConfiguration(), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        await runWindowLaunchTask(store)
        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.health?.startup?.launchId, priorHealth.startup?.launchId, "adopted, not relaunched")
        try await pollUntil("live stats open") { store.connectionState == .open }

        // A window reopens: the same task runs again.
        await runWindowLaunchTask(store)

        XCTAssertEqual(
            store.daemonState, .running,
            "a second launch task against an adopted healthy daemon must not degrade"
        )
        XCTAssertEqual(store.startupPhase, .ready)
        XCTAssertEqual(store.health?.startup?.pid, priorHealth.startup?.pid)
        XCTAssertEqual(store.health?.startup?.launchId, priorHealth.startup?.launchId)
        XCTAssertEqual(daemon.spawns().count, 1, "the adopted daemon is the only one")
        XCTAssertTrue(prior.isRunning)
        let restores = await fans.restores
        XCTAssertEqual(restores, 0)
        try await pollUntil("live stats still open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }

    // MARK: A server that is not ours on the port

    /// The app's daemon is gone and a stranger answers on its port. A
    /// healthy answer from the stranger must never read as this app's engine
    /// running. The app lets its lost daemon go and stays Degraded naming
    /// the other server, rather than Stopped, which hid the cause. Stop
    /// never signals the stranger and the configured port is not moved.
    @MainActor
    func testAnotherServerOnThePortIsNamedAndNeverSignalled() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let prior = try daemon.launchOutsideTheApp(launchID: "prior-session-\(UUID().uuidString)")
        addTeardownBlock { if prior.isRunning { prior.terminate() } }
        _ = try await daemon.waitUntilHealthy()
        let fans = FanCallRecorder()
        let supervisor = DaemonSupervisor()
        let store = daemon.makeStore(
            configuration: daemon.configuration(fanMode: .default),
            supervisor: supervisor,
            fans: fans
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)

        // The adopted daemon goes away and `mtplx serve`, started in a
        // terminal, takes its port.
        prior.terminate()
        prior.waitUntilExit()
        let stranger = try daemon.launchOutsideTheApp(launchID: nil)
        addTeardownBlock { if stranger.isRunning { stranger.terminate() } }
        let strangerHealth = try await daemon.waitUntilHealthy()
        XCTAssertNil(strangerHealth.startup?.launchId)

        await store.refresh()
        await store.awaitDaemonTeardown()
        // The supervisor's terminal snapshot reaches the store on its own hop.
        try await pollUntil("the lost daemon is released") {
            !supervisor.isRunning() && store.daemonState != .stopping
        }

        let strangerPID = try XCTUnwrap(strangerHealth.startup?.pid)
        guard case .degraded(let reason) = store.daemonState else {
            return XCTFail("another server on the port must stay visible as Degraded, got \(store.daemonState)")
        }
        XCTAssertEqual(
            reason,
            "Another MTPLX server holds port \(daemon.port) (pid \(strangerPID)); this app is not connected to it."
        )
        let named = badge(store)
        XCTAssertEqual(named.tone, .failed)
        XCTAssertTrue(named.label.hasPrefix("Degraded — Another MTPLX server"), named.label)
        XCTAssertEqual(named.help, "Degraded: \(reason)")
        XCTAssertNotEqual(
            store.health?.startup?.pid, strangerPID,
            "a stranger's pid must never become the app's, where Stop would signal it"
        )
        XCTAssertFalse(supervisor.isRunning(), "the app no longer claims a daemon it lost")

        await store.stopDaemon()
        XCTAssertTrue(stranger.isRunning, "Stop never signals the other server")
        let stillThere = try await daemon.waitUntilHealthy()
        XCTAssertEqual(stillThere.startup?.pid, strangerPID)
        XCTAssertEqual(store.configuration.port, daemon.port)
    }

    /// An adopted daemon exits and another server takes its port. The live
    /// stats stream ends cleanly and reopens on the new server, and the
    /// watchdog keeps probing. Neither may let the other server's answer
    /// stand for the app's daemon (on 1de2b1c0 the watchdog published it as
    /// `health`, and Stop then signalled its pid).
    @MainActor
    private func assertAnotherServerAfterACleanStreamEnd(strangerLaunchID: String?) async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let prior = try daemon.launchOutsideTheApp(launchID: "prior-session-\(UUID().uuidString)")
        addTeardownBlock { if prior.isRunning { prior.terminate() } }
        let priorHealth = try await daemon.waitUntilHealthy()
        try daemon.settingsStore.save(daemon.configuration(fanMode: .default))
        let supervisor = DaemonSupervisor()
        let store = daemon.makeStore(
            configuration: MTPLXAppConfiguration(),
            supervisor: supervisor,
            fans: FanCallRecorder()
        )
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        store.loadPersistedSettings()
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.health?.startup?.launchId, priorHealth.startup?.launchId, "adopted")
        try await pollUntil("live stats open") { store.connectionState == .open }

        prior.terminate()
        prior.waitUntilExit()
        let stranger = try daemon.launchOutsideTheApp(launchID: strangerLaunchID)
        addTeardownBlock { if stranger.isRunning { stranger.terminate() } }
        let strangerHealth = try await daemon.waitUntilHealthy()
        let strangerPID = try XCTUnwrap(strangerHealth.startup?.pid)
        XCTAssertEqual(strangerHealth.startup?.launchId, strangerLaunchID)

        try await pollUntil("the other server is named", timeout: 15) {
            if case .degraded(let reason) = store.daemonState {
                return reason.hasPrefix("Another MTPLX server holds port \(daemon.port)")
            }
            return false
        }
        // More watchdog rounds (every 3 s) before Stop: nothing may publish
        // the other server's answer as the app's daemon meanwhile.
        try await Task.sleep(nanoseconds: 4_000_000_000)
        guard case .degraded(let reason) = store.daemonState else {
            return XCTFail("expected Degraded naming the other server, got \(store.daemonState)")
        }
        XCTAssertTrue(reason.contains("(pid \(strangerPID))"), reason)
        XCTAssertNotEqual(store.health?.startup?.pid, strangerPID)
        XCTAssertTrue(badge(store).label.hasPrefix("Degraded — Another MTPLX server"), badge(store).label)
        XCTAssertFalse(supervisor.isRunning(), "the app let go of the daemon it lost")

        await store.stopDaemon()
        XCTAssertTrue(stranger.isRunning, "Stop must never signal another server")
        let stillThere = try await daemon.waitUntilHealthy()
        XCTAssertEqual(stillThere.startup?.pid, strangerPID)
        XCTAssertEqual(daemon.spawns().count, 2, "the prior daemon and the stranger; the app launched nothing")
    }

    @MainActor
    func testAnotherServerWithoutALaunchIDAfterACleanStreamEnd() async throws {
        try await assertAnotherServerAfterACleanStreamEnd(strangerLaunchID: nil)
    }

    @MainActor
    func testAnotherServerWithADifferentLaunchIDAfterACleanStreamEnd() async throws {
        try await assertAnotherServerAfterACleanStreamEnd(strangerLaunchID: "other-session-\(UUID().uuidString)")
    }

    // MARK: A launch on a fallback port, then the window reopens

    /// The configured port is taken by `mtplx serve` from a terminal, so the
    /// app launches on the next free port and settings keep the configured
    /// one (#503). Reopening the window reloads settings from disk. Every
    /// later request must still reach the daemon where it listens, not the
    /// reloaded port, where the other server answers.
    @MainActor
    func testFallbackPortDaemonStaysReachableAfterTheWindowReopens() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let occupant = try daemon.launchOutsideTheApp(launchID: nil)
        addTeardownBlock { if occupant.isRunning { occupant.terminate() } }
        let occupantHealth = try await daemon.waitUntilHealthy()
        try daemon.settingsStore.save(daemon.configuration(fanMode: .default))
        let store = daemon.makeStore(configuration: MTPLXAppConfiguration(), fans: FanCallRecorder())
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        store.loadPersistedSettings()
        await store.startDaemon()
        XCTAssertEqual(store.daemonState, .running)
        let fallbackPort = try XCTUnwrap(store.baseURL.port)
        XCTAssertNotEqual(fallbackPort, daemon.port, "the launch moved off the occupied port")
        XCTAssertNotNil(store.portFallbackNotice)
        let launched = try XCTUnwrap(store.health?.startup)
        try await pollUntil("live stats open") { store.connectionState == .open }

        // The window reopens: its launch task reloads settings and starts.
        store.loadPersistedSettings()
        XCTAssertEqual(store.configuration.port, daemon.port, "settings keep the configured port")
        await store.startDaemon()

        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.baseURL.port, fallbackPort)
        XCTAssertEqual(
            store.apiClient.baseURL.port, fallbackPort,
            "requests go where the daemon listens, not to the reloaded port"
        )
        let answer = try await store.apiClient.health()
        XCTAssertEqual(answer.startup?.launchId, launched.launchId, "the app's daemon answers, not the occupant")
        XCTAssertNotEqual(answer.startup?.pid, occupantHealth.startup?.pid)
        XCTAssertEqual(daemon.spawns().count, 2, "the occupant and one app daemon")
        XCTAssertNotNil(store.portFallbackNotice, "the fallback banner stays while that daemon runs")
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")

        await store.stopDaemon()
        XCTAssertTrue(occupant.isRunning, "Stop never signals the server on the configured port")
    }

    // MARK: Live stats drop while the daemon stays healthy

    @MainActor
    func testLiveStatsDropWithHealthyDaemonShowsLiveChannelStateNotDegraded() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let fans = FanCallRecorder()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        try await pollUntil("live stats open") { store.connectionState == .open }
        let launched = try XCTUnwrap(store.health?.startup)

        // The Mac sleeps: the stream drops while /health stays healthy.
        try daemon.setStreamDown(true)
        try await pollUntil("live stats reconnecting") {
            if case .reconnecting = store.connectionState { return true }
            return false
        }
        // On wake the window's launch task runs again.
        await store.startDaemon()

        XCTAssertEqual(store.daemonState, .running, "a dropped stats stream is not an engine failure")
        let dropped = badge(store)
        XCTAssertEqual(dropped.tone, .pending)
        XCTAssertTrue(dropped.label.hasPrefix("Running · "), "badge read \(dropped.label)")
        XCTAssertEqual(daemon.spawns().count, 1)

        // The stream comes back: Running again, same daemon, no restart.
        try daemon.setStreamDown(false)
        try await pollUntil("live stats open again", timeout: 15) { store.connectionState == .open }
        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.health?.startup?.launchId, launched.launchId)
        XCTAssertEqual(daemon.spawns().count, 1)
        XCTAssertEqual(badge(store).label, "Running")
    }

    // MARK: A transient /health failure

    @MainActor
    func testTransientHealthFailureRecoversWithoutRestart() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let fans = FanCallRecorder()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }
        await store.startDaemon()
        try await pollUntil("live stats open") { store.connectionState == .open }
        let launched = try XCTUnwrap(store.health?.startup)

        // /health stops answering for a few seconds (a long commit, #487)
        // just as the benchmark asks for a ready daemon.
        try daemon.setHealthDown(true)
        let answersAgain = Task {
            try? await Task.sleep(nanoseconds: 6_000_000_000)
            try? daemon.setHealthDown(false)
        }
        do {
            _ = try await store.ensureDaemonReadyForBenchmark()
            XCTFail("a daemon that did not answer /health was reported ready")
        } catch BenchmarkDaemonReadinessError.startupFailed(let reason) {
            XCTFail("a busy daemon was reported as a failed start: \(reason)")
        } catch {
            // Unreachable for now: true, it did not answer this time.
        }
        XCTAssertEqual(store.daemonState, .running, "a daemon that is slow to answer is busy, not degraded")

        // It answers again and the user presses Refresh.
        answersAgain.cancel()
        try daemon.setHealthDown(false)
        await store.refresh()

        XCTAssertEqual(store.daemonState, .running)
        XCTAssertEqual(store.startupPhase, .ready)
        XCTAssertEqual(store.health?.startup?.launchId, launched.launchId)
        XCTAssertEqual(store.health?.startup?.pid, launched.pid)
        XCTAssertEqual(daemon.spawns().count, 1, "recovered without a restart")
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).tone, .healthy)
    }

    // MARK: Two starts at once

    @MainActor
    func testStartWhileAStartIsLoadingJoinsIt() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let fans = FanCallRecorder()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        // The model is still loading: the port is open, /health is not ready.
        try daemon.setHealthDown(true)
        let first = Task { @MainActor in await store.startDaemon() }
        try await pollUntil("daemon launched") { daemon.spawns().count == 1 }
        // Start pressed again, or the window task ran again, mid-load.
        let second = Task { @MainActor in await store.startDaemon() }
        try await Task.sleep(nanoseconds: 1_000_000_000)
        try daemon.setHealthDown(false)
        await first.value
        await second.value

        XCTAssertEqual(store.daemonState, .running, "a start during a load must not degrade either start")
        XCTAssertEqual(store.startupPhase, .ready)
        let spawns = daemon.spawns()
        XCTAssertEqual(spawns.count, 1, "one daemon for both requests")
        XCTAssertEqual(store.health?.startup?.pid, spawns.first?.pid, "the loading daemon was kept, not replaced")
        XCTAssertEqual(store.health?.startup?.launchId, spawns.first?.launchID)
        try await pollUntil("live stats open") { store.connectionState == .open }
        let logged = await logsMention(store, "joining it")
        XCTAssertTrue(logged, "the Logs window records that the second request joined the first")
    }

    // MARK: Start after Stop

    /// A store whose first start is suspended in the supervisor's first
    /// /health probe, before it reserves a process.
    @MainActor
    private func storeWithAStartSuspendedInItsFirstProbe() async throws -> (
        daemon: ReconnectFakeDaemon,
        store: MTPLXBackendStore,
        gate: FirstCallGate,
        first: Task<Void, Never>
    ) {
        let daemon = try ReconnectFakeDaemon.make()
        let gate = FirstCallGate()
        let supervisor = DaemonSupervisor(initialHealthProbe: { url, apiKey in
            await gate.pass()
            return await DaemonSupervisor.defaultHealthProbe(url, apiKey)
        })
        let store = daemon.makeStore(
            configuration: daemon.configuration(fanMode: .default),
            supervisor: supervisor,
            fans: FanCallRecorder()
        )
        let first = Task { @MainActor in await store.startDaemon() }
        await gate.waitUntilEntered()
        XCTAssertEqual(daemon.spawns().count, 0)
        return (daemon, store, gate, first)
    }

    /// Stop overtakes a start that is still in its first probe, then Start
    /// is pressed. The second request must launch: joining the cancelled
    /// start ended it with that start's cancellation and launched nothing.
    @MainActor
    func testStartAfterStopDuringASuspendedStartLaunches() async throws {
        let (daemon, store, gate, first) = try await storeWithAStartSuspendedInItsFirstProbe()
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        await store.stopDaemon()
        let second = Task { @MainActor in await store.startDaemon() }
        try await Task.sleep(nanoseconds: 300_000_000)
        await gate.open()
        await first.value
        await second.value

        XCTAssertEqual(store.daemonState, .running, "Start after Stop must launch")
        XCTAssertEqual(store.startupPhase, .ready)
        let spawns = daemon.spawns()
        XCTAssertEqual(spawns.count, 1)
        XCTAssertEqual(store.health?.startup?.launchId, spawns.first?.launchID)
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }

    /// The same through Restart, which is Stop then Start on one task.
    @MainActor
    func testRestartDuringASuspendedStartLaunches() async throws {
        let (daemon, store, gate, first) = try await storeWithAStartSuspendedInItsFirstProbe()
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        let restart = Task { @MainActor in
            await store.stopDaemon()
            await store.startDaemon()
        }
        try await Task.sleep(nanoseconds: 300_000_000)
        await gate.open()
        await first.value
        await restart.value

        XCTAssertEqual(store.daemonState, .running, "Restart during a start must end running")
        let spawns = daemon.spawns()
        XCTAssertEqual(spawns.count, 1)
        XCTAssertEqual(store.health?.startup?.launchId, spawns.first?.launchID)
        try await pollUntil("live stats open") { store.connectionState == .open }
    }

    /// The first caller is cancelled (its window closed) while a second
    /// request waits on the same start. Neither cancels the load.
    @MainActor
    func testCancellingTheFirstCallerWhileAnotherJoinsKeepsTheLoad() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let store = daemon.makeStore(configuration: daemon.configuration(fanMode: .default), fans: FanCallRecorder())
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        try daemon.setHealthDown(true)
        let first = Task { @MainActor in await store.startDaemon() }
        try await pollUntil("daemon launched") { daemon.spawns().count == 1 }
        let second = Task { @MainActor in await store.startDaemon() }
        try await Task.sleep(nanoseconds: 300_000_000)
        first.cancel()
        try await Task.sleep(nanoseconds: 300_000_000)
        try daemon.setHealthDown(false)
        await second.value
        await first.value

        XCTAssertEqual(store.daemonState, .running)
        let spawns = daemon.spawns()
        XCTAssertEqual(spawns.count, 1)
        XCTAssertEqual(store.health?.startup?.pid, spawns.first?.pid)
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }

    // MARK: Closing the window during a model load

    /// Closing the main window cancels its launch task. The start runs on a
    /// task the store owns, so the load carries on and the engine comes up.
    /// Before, the cancellation reached the supervisor's health wait, which
    /// stopped the loading daemon: "Degraded — CancellationError()".
    @MainActor
    func testClosingTheWindowDuringALoadStillReachesRunning() async throws {
        let daemon = try ReconnectFakeDaemon.make()
        let fans = FanCallRecorder()
        try daemon.settingsStore.save(daemon.configuration(fanMode: .default))
        let store = daemon.makeStore(configuration: MTPLXAppConfiguration(), fans: fans)
        addTeardownBlock { @MainActor in await store.stopDaemon() }

        // The model is still loading: the port is open, /health is not ready.
        try daemon.setHealthDown(true)
        // The daemon half of the window's launch task.
        let window = Task { @MainActor in
            store.loadPersistedSettings()
            await store.startDaemon()
        }
        try await pollUntil("daemon launched") { daemon.spawns().count == 1 }
        try await Task.sleep(nanoseconds: 500_000_000)
        // The user closes the window.
        window.cancel()
        try await Task.sleep(nanoseconds: 500_000_000)
        try daemon.setHealthDown(false)

        try await pollUntil("the load finished", timeout: 10) { store.daemonState == .running }
        await window.value
        XCTAssertEqual(store.startupPhase, .ready)
        let spawns = daemon.spawns()
        XCTAssertEqual(spawns.count, 1, "the loading daemon was kept, not relaunched")
        XCTAssertEqual(store.health?.startup?.pid, spawns.first?.pid)
        try await pollUntil("live stats open") { store.connectionState == .open }
        XCTAssertEqual(badge(store).label, "Running")
    }
}
