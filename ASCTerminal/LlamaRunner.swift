import Foundation
import llama

final class LlamaRunner: @unchecked Sendable {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var sampler: UnsafeMutablePointer<llama_sampler>
    private let lock = NSLock()

    init(modelPath: String) throws {
        llama_backend_init()
        var mp = llama_model_default_params()
        #if targetEnvironment(simulator)
        mp.n_gpu_layers = 0
        #endif
        guard let model = llama_model_load_from_file(modelPath, mp) else { throw RunnerError.modelLoadFailed }
        var cp = llama_context_default_params()
        cp.n_ctx = 2048
        cp.n_batch = 512
        let threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        cp.n_threads = Int32(threads)
        cp.n_threads_batch = Int32(threads)
        guard let context = llama_init_from_model(model, cp) else {
            llama_model_free(model); throw RunnerError.contextFailed
        }
        guard let vocab = llama_model_get_vocab(model) else {
            llama_free(context); llama_model_free(model); throw RunnerError.contextFailed
        }
        var sp = llama_sampler_chain_default_params()
        guard let sampler = llama_sampler_chain_init(sp) else {
            llama_free(context); llama_model_free(model); throw RunnerError.contextFailed
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40))
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9, 1))
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.3))
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234))
        self.model = model; self.context = context; self.vocab = vocab; self.sampler = sampler
    }

    deinit {
        llama_sampler_free(sampler)
        llama_free(context)
        llama_model_free(model)
        llama_backend_free()
    }

    func respond(system: String, user: String, history: [(String, String)]) async throws -> String {
        try await Task.detached { [self] in
            lock.lock(); defer { lock.unlock() }
            var messages: [llama_chat_message] = []
            var storage: [Data] = []
            func add(_ role: String, _ content: String) {
                let rd = role.data(using: .utf8)!; let cd = content.data(using: .utf8)!
                storage.append(rd + [0]); storage.append(cd + [0])
                messages.append(llama_chat_message(role: storage[storage.count - 2].withUnsafeBytes { $0.bindMemory(to: CChar.self).baseAddress }, content: storage[storage.count - 1].withUnsafeBytes { $0.bindMemory(to: CChar.self).baseAddress }))
            }
            add("system", system)
            for (u, a) in history { add("user", u); add("assistant", a) }
            add("user", user)
            guard let tmpl = llama_model_chat_template(model, nil) else { throw RunnerError.noChatTemplate }
            var formatted = [CChar](repeating: 0, count: 8192)
            let count = llama_chat_apply_template(tmpl, &messages, messages.count, true, &formatted, Int32(formatted.count))
            guard count >= 0 else { throw RunnerError.templateFailed }
            let prompt = String(bytes: formatted.prefix(Int(count)).map { UInt8(bitPattern: $0) }, encoding: .utf8) ?? ""
            let tokens = try tokenize(prompt)
            var batch = llama_batch_init(512, 0, 1); defer { llama_batch_free(batch) }
            guard tokens.count < 512 else { throw RunnerError.promptTooLong }
            for (i, token) in tokens.enumerated() {
                batch.token[i] = token; batch.pos[i] = Int32(i); batch.n_seq_id[i] = 1; batch.seq_id[i]![0] = 0; batch.logits[i] = 0
            }
            batch.logits[tokens.count - 1] = 1; batch.n_tokens = Int32(tokens.count)
            llama_memory_clear(llama_get_memory(context), true)
            guard llama_decode(context, batch) == 0 else { throw RunnerError.decodeFailed }
            var output = ""
            var pos = Int32(tokens.count)
            for _ in 0..<384 {
                let token = llama_sampler_sample(sampler, context, -1)
                if llama_vocab_is_eog(vocab, token) { break }
                output += piece(token)
                batch.n_tokens = 0
                batch.token[0] = token; batch.pos[0] = pos; batch.n_seq_id[0] = 1; batch.seq_id[0]![0] = 0; batch.logits[0] = 1; batch.n_tokens = 1
                guard llama_decode(context, batch) == 0 else { throw RunnerError.decodeFailed }
                pos += 1
            }
            return output.trimmingCharacters(in: .whitespacesAndNewlines)
        }.value
    }

    private func tokenize(_ text: String) throws -> [llama_token] {
        let bytes = Array(text.utf8)
        var tokens = [llama_token](repeating: 0, count: bytes.count + 8)
        let count = bytes.withUnsafeBytes { ptr -> Int32 in
            llama_tokenize(vocab, ptr.bindMemory(to: CChar.self).baseAddress, Int32(bytes.count), &tokens, Int32(tokens.count), true, false)
        }
        guard count >= 0 else { throw RunnerError.tokenizeFailed }
        return Array(tokens.prefix(Int(count)))
    }

    private func piece(_ token: llama_token) -> String {
        var buffer = [CChar](repeating: 0, count: 128)
        var n = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, false)
        if n < 0 { buffer = [CChar](repeating: 0, count: Int(-n) + 1); n = llama_token_to_piece(vocab, token, &buffer, Int32(buffer.count), 0, false) }
        return String(bytes: buffer.prefix(max(0, Int(n))).map { UInt8(bitPattern: $0) }, encoding: .utf8) ?? ""
    }

    enum RunnerError: LocalizedError {
        case modelLoadFailed, contextFailed, noChatTemplate, templateFailed, promptTooLong, decodeFailed, tokenizeFailed
        var errorDescription: String? {
            switch self { case .modelLoadFailed: return "Could not load GGUF model."; case .contextFailed: return "Could not initialize local AI context."; case .noChatTemplate: return "Model has no supported chat template."; case .templateFailed: return "Could not format the chat prompt."; case .promptTooLong: return "Prompt is too long for this terminal context."; case .decodeFailed: return "Local inference failed."; case .tokenizeFailed: return "Tokenization failed." }
        }
    }
}
