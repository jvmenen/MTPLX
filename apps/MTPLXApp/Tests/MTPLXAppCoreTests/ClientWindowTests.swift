import XCTest

@testable import MTPLXAppCore

/// Pi and OpenCode are configured from the window the daemon executes, with
/// the answer ceiling priced as half of it. On 2026-09-29 the app wrote
/// 262,144 into Pi's contextWindow and maxTokens from Settings while the
/// engine could not serve a conversation that long.
final class ClientWindowTests: XCTestCase {
    private let flashNext = "/models/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed"
    private let qwen36 = "/models/Qwen3.6-27B-MTPLX-Optimized-Speed"

    // MARK: Budget

    func testServedWindowWinsOverTheSettingAtEveryBoundary() {
        let settings = MTPLXAppConfiguration(model: flashNext, contextWindow: 262_144)
        for (tokens, answer) in [(4_096, 2_048), (32_768, 16_384), (98_304, 49_152), (262_144, 131_072)] {
            let budget = ClientContextBudget.resolve(
                configuration: settings,
                served: ServedExecutionWindow(tokens: tokens, answerTokens: answer),
                defaultWindow: 131_072
            )
            XCTAssertEqual(budget, ClientContextBudget(contextWindow: tokens, answerTokens: answer))
        }
    }

    func testAnswerShareIsHalfTheWindowWhenTheServerGivesNone() {
        let settings = MTPLXAppConfiguration(model: flashNext, contextWindow: 262_144)
        let missing = ClientContextBudget.resolve(
            configuration: settings,
            served: ServedExecutionWindow(tokens: 65_536, answerTokens: nil),
            defaultWindow: 131_072
        )
        XCTAssertEqual(missing.answerTokens, 32_768)
        // An answer share larger than the window is never advertised.
        let oversized = ClientContextBudget.resolve(
            configuration: settings,
            served: ServedExecutionWindow(tokens: 8_192, answerTokens: 50_000),
            defaultWindow: 131_072
        )
        XCTAssertEqual(oversized, ClientContextBudget(contextWindow: 8_192, answerTokens: 8_192))
        XCTAssertEqual(ClientContextBudget.answerTokens(forWindow: 1), 1)
    }

    func testWithoutAServedWindowTheSettingIsUsed() {
        let settings = MTPLXAppConfiguration(model: qwen36, contextWindow: 65_536)
        let budget = ClientContextBudget.resolve(configuration: settings, served: nil, defaultWindow: 131_072)
        XCTAssertEqual(budget, ClientContextBudget(contextWindow: 65_536, answerTokens: 32_768))
        let zero = ClientContextBudget.resolve(
            configuration: settings,
            served: ServedExecutionWindow(tokens: 0, answerTokens: 0),
            defaultWindow: 131_072
        )
        XCTAssertEqual(zero, budget)
    }

    func testHealthDecodesTheExecutionWindowAndToleratesItsAbsence() throws {
        let with = try JSONDecoder().decode(HealthPayload.self, from: Data(Self.health(window: """
        ,"execution_window": {"tokens": 98304, "answer_tokens": 49152, "basis": "machine_fit"}
        """).utf8))
        XCTAssertEqual(with.executionWindow, ServedExecutionWindow(tokens: 98_304, answerTokens: 49_152, basis: "machine_fit"))
        let without = try JSONDecoder().decode(HealthPayload.self, from: Data(Self.health(window: "").utf8))
        XCTAssertNil(without.executionWindow)
    }

    // MARK: Pi

    func testPiAdvertisesHalfTheWindowAsItsAnswerCeiling() throws {
        let url = temporaryDirectory().appendingPathComponent("models.json")
        _ = try PiIntegration(configURL: url).sync(
            configuration: MTPLXAppConfiguration(model: qwen36, port: 8000, contextWindow: nil)
        )
        let model = try piModel(at: url)
        XCTAssertEqual(model["contextWindow"]?.intValue, 131_072)
        XCTAssertEqual(model["maxTokens"]?.intValue, 65_536)
    }

    func testPiIsConfiguredFromTheServedWindowAtBoundaryValues() throws {
        for (tokens, answer) in [(4_096, 2_048), (32_768, 16_384), (262_144, 131_072)] {
            let url = temporaryDirectory().appendingPathComponent("models.json")
            _ = try PiIntegration(configURL: url).sync(
                configuration: MTPLXAppConfiguration(model: flashNext, port: 8000, contextWindow: 262_144),
                servedWindow: ServedExecutionWindow(tokens: tokens, answerTokens: answer)
            )
            let model = try piModel(at: url)
            XCTAssertEqual(model["contextWindow"]?.intValue, tokens)
            XCTAssertEqual(model["maxTokens"]?.intValue, answer)
        }
    }

    func testPiRefreshesTheWindowMTPLXWroteAndKeepsAUserEdit() throws {
        let url = temporaryDirectory().appendingPathComponent("models.json")
        let integration = PiIntegration(configURL: url)
        let configuration = MTPLXAppConfiguration(model: flashNext, port: 8000, contextWindow: 262_144)
        let served = ServedExecutionWindow(tokens: 98_304, answerTokens: 49_152)

        // What every earlier MTPLX wrote: maxTokens equal to the window.
        try writePiModel(at: url, contextWindow: 262_144, maxTokens: 262_144)
        // The write before the daemon starts leaves the window fields alone.
        _ = try integration.sync(configuration: configuration)
        XCTAssertEqual(try piModel(at: url)["contextWindow"]?.intValue, 262_144)
        // Once the daemon publishes its window, MTPLX's pair follows it.
        let refreshed = try integration.sync(configuration: configuration, servedWindow: served)
        XCTAssertTrue(refreshed.didChange)
        var model = try piModel(at: url)
        XCTAssertEqual(model["contextWindow"]?.intValue, 98_304)
        XCTAssertEqual(model["maxTokens"]?.intValue, 49_152)
        XCTAssertFalse(try integration.sync(configuration: configuration, servedWindow: served).didChange)
        // A later daemon with a larger window moves MTPLX's own pair again.
        _ = try integration.sync(
            configuration: configuration,
            servedWindow: ServedExecutionWindow(tokens: 262_144, answerTokens: 131_072)
        )
        model = try piModel(at: url)
        XCTAssertEqual(model["contextWindow"]?.intValue, 262_144)
        XCTAssertEqual(model["maxTokens"]?.intValue, 131_072)

        // A pair the user chose is theirs (#282).
        try writePiModel(at: url, contextWindow: 65_536, maxTokens: 20_000)
        _ = try integration.sync(configuration: configuration, servedWindow: served)
        model = try piModel(at: url)
        XCTAssertEqual(model["contextWindow"]?.intValue, 65_536)
        XCTAssertEqual(model["maxTokens"]?.intValue, 20_000)
    }

    func testOwnershipSignatureRecognisesOnlyMTPLXPairs() {
        func entry(_ window: Int, _ maxTokens: Int) -> [String: JSONValue] {
            ["contextWindow": .number(Double(window)), "maxTokens": .number(Double(maxTokens))]
        }
        XCTAssertTrue(PiIntegration.windowFieldsWereWrittenByMTPLX(entry(262_144, 262_144)))
        XCTAssertTrue(PiIntegration.windowFieldsWereWrittenByMTPLX(entry(98_304, 49_152)))
        XCTAssertFalse(PiIntegration.windowFieldsWereWrittenByMTPLX(entry(131_072, 20_000)))
        XCTAssertFalse(PiIntegration.windowFieldsWereWrittenByMTPLX(["contextWindow": .number(8_192)]))
    }

    // MARK: OpenCode

    func testOpenCodeLimitsFollowTheServedWindowAtBoundaryValues() throws {
        for (tokens, output) in [(32_768, 16_384), (40_000, 20_000), (262_144, 32_000)] {
            let url = temporaryDirectory().appendingPathComponent("opencode.json")
            _ = try openCode(url).sync(
                configuration: MTPLXAppConfiguration(model: flashNext, port: 8000, contextWindow: 262_144),
                servedWindow: ServedExecutionWindow(
                    tokens: tokens,
                    answerTokens: ClientContextBudget.answerTokens(forWindow: tokens)
                )
            )
            let limit = try openCodeLimit(at: url)
            XCTAssertEqual(limit["context"]?.intValue, tokens)
            XCTAssertEqual(limit["output"]?.intValue, output)
        }
    }

    func testOpenCodeKeepsItsLimitsUntilTheDaemonPublishesAWindow() throws {
        let url = temporaryDirectory().appendingPathComponent("opencode.json")
        let integration = openCode(url)
        let configuration = MTPLXAppConfiguration(model: flashNext, port: 8000, contextWindow: 262_144)
        // First launch, nothing written yet: the setting.
        _ = try integration.sync(configuration: configuration)
        XCTAssertEqual(try openCodeLimit(at: url)["context"]?.intValue, 262_144)
        // The daemon executes less: the limits follow it.
        let served = ServedExecutionWindow(tokens: 98_304, answerTokens: 49_152)
        XCTAssertTrue(try integration.sync(configuration: configuration, servedWindow: served).didChange)
        XCTAssertEqual(try openCodeLimit(at: url)["context"]?.intValue, 98_304)
        XCTAssertEqual(try openCodeLimit(at: url)["output"]?.intValue, 32_000)
        // The next launch's first write keeps them instead of flipping back.
        XCTAssertFalse(try integration.sync(configuration: configuration).didChange)
        XCTAssertEqual(try openCodeLimit(at: url)["context"]?.intValue, 98_304)
    }

    // MARK: Helpers

    private func openCode(_ url: URL) -> OpenCodeIntegration {
        OpenCodeIntegration(
            configURL: url,
            desktopSettingsStoreURL: url.deletingLastPathComponent().appendingPathComponent("default.dat")
        )
    }

    private func temporaryDirectory() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-client-window-tests-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func piModel(at url: URL) throws -> [String: JSONValue] {
        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        let models = try XCTUnwrap(
            root["providers"]?.objectValue?["mtplx"]?.objectValue?["models"]?.arrayValue
        )
        return try XCTUnwrap(models.first?.objectValue)
    }

    private func writePiModel(at url: URL, contextWindow: Int, maxTokens: Int) throws {
        let modelID = PiIntegration.modelID(for: flashNext)
        let root: [String: JSONValue] = [
            "providers": .object([
                "mtplx": .object([
                    "baseUrl": .string("http://127.0.0.1:8000/v1"),
                    "models": .array([
                        .object([
                            "id": .string(modelID),
                            "contextWindow": .number(Double(contextWindow)),
                            "maxTokens": .number(Double(maxTokens)),
                        ]),
                    ]),
                ]),
            ]),
        ]
        try JSONEncoder().encode(root).write(to: url)
    }

    private func openCodeLimit(at url: URL) throws -> [String: JSONValue] {
        let root = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: url))
        let models = try XCTUnwrap(root["provider"]?.objectValue?["mtplx"]?.objectValue?["models"]?.objectValue)
        let model = try XCTUnwrap(models.values.first?.objectValue)
        return try XCTUnwrap(model["limit"]?.objectValue)
    }

    private static func health(window: String) -> String {
        """
        {"ok": true, "model": "m", "model_path": "/m", "generation_mode": "mtp",
         "load_mtp": true, "mtp_enabled": true, "depth": 3, "profile": {},
         "context_window": 262144, "active_requests": 0, "reasoning_parser": "qwen3"\(window)}
        """
    }
}

private extension JSONValue {
    var objectValue: [String: JSONValue]? {
        if case .object(let value) = self { return value }
        return nil
    }

    var arrayValue: [JSONValue]? {
        if case .array(let value) = self { return value }
        return nil
    }
}
