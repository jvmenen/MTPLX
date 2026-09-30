import Foundation

/// The window and answer ceiling MTPLX writes into Pi and OpenCode.
///
/// On 2026-09-29 the app wrote 262,144 into Pi's `contextWindow` and
/// `maxTokens` straight from Settings while the engine could not serve a
/// conversation that long. Once the daemon is up it publishes the window it
/// executes (`/health` `execution_window`), and clients are configured from
/// that. Before the daemon answers (the first write of a launch) the setting
/// is used, and the same answer rule applies.
public struct ClientContextBudget: Equatable, Sendable {
    /// Conversation length, prompt plus answer, the client should plan for.
    public var contextWindow: Int
    /// The longest answer the client should ask for inside that window.
    public var answerTokens: Int

    public init(contextWindow: Int, answerTokens: Int) {
        self.contextWindow = contextWindow
        self.answerTokens = answerTokens
    }

    /// Half the window: the rule the server publishes
    /// (mtplx/server/served_window.py `answer_share_tokens`). A client that
    /// advertises the whole window as its output ceiling asks for an answer
    /// as long as the conversation while the history is still short, and
    /// leaves the history no room once that answer arrives. Half still allows
    /// 131,072-token answers on a 262,144 window, and equals the 16,384 Pi
    /// assumes on a 32,768 window; clients clamp every request to the room
    /// their prompt leaves, so prompt plus answer stays inside the window.
    public static func answerTokens(forWindow window: Int) -> Int {
        max(1, window / 2)
    }

    /// The served window when the daemon published one, else the setting
    /// (`configuration.effectiveContextWindow(default:)`).
    public static func resolve(
        configuration: MTPLXAppConfiguration,
        served: ServedExecutionWindow?,
        defaultWindow: Int
    ) -> ClientContextBudget {
        if let served, served.tokens > 0 {
            let answer = served.answerTokens.flatMap { $0 > 0 ? $0 : nil }
                ?? answerTokens(forWindow: served.tokens)
            return ClientContextBudget(
                contextWindow: served.tokens,
                answerTokens: min(answer, served.tokens)
            )
        }
        let window = configuration.effectiveContextWindow(default: defaultWindow)
        return ClientContextBudget(
            contextWindow: window,
            answerTokens: answerTokens(forWindow: window)
        )
    }
}
