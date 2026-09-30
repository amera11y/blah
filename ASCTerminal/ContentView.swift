import SwiftUI
import UniformTypeIdentifiers

struct TerminalLine: Identifiable {
    let id = UUID()
    let prompt: String
    let text: String
}

@MainActor
final class TerminalViewModel: ObservableObject {
    @Published var lines: [TerminalLine] = [
        TerminalLine(prompt: "", text: "ASC TERMINAL — LOCAL AI"),
        TerminalLine(prompt: "", text: "ASCII INFUSED • OFFLINE INFERENCE"),
        TerminalLine(prompt: "", text: "Import a GGUF model to begin.")
    ]
    @Published var input = ""
    @Published var modelName = "No model loaded"
    @Published var isLoading = false
    @Published var error: String?
    private var runner: LlamaRunner?
    private var history: [(String, String)] = []

    func importModel(url: URL) {
        do {
            let accessed = url.startAccessingSecurityScopedResource()
            defer { if accessed { url.stopAccessingSecurityScopedResource() } }
            let models = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            let dir = models[0].appendingPathComponent("Models", isDirectory: true)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let destination = dir.appendingPathComponent(url.lastPathComponent)
            if destination.path != url.path {
                if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
                try FileManager.default.copyItem(at: url, to: destination)
            }
            runner = try LlamaRunner(modelPath: destination.path)
            modelName = url.lastPathComponent
            lines.append(TerminalLine(prompt: "", text: "Model loaded: \(modelName)"))
        } catch {
            self.error = error.localizedDescription
        }
    }

    func submit() {
        let command = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !command.isEmpty else { return }
        input = ""
        lines.append(TerminalLine(prompt: "ASC >>", text: command))
        if ["exit", "quit"].contains(command.lowercased()) {
            lines.append(TerminalLine(prompt: "", text: "Session remains open on iPhone. Type another command to continue."))
            return
        }
        guard let runner else {
            lines.append(TerminalLine(prompt: "", text: "No model loaded. Use Import GGUF first."))
            return
        }
        isLoading = true
        Task {
            do {
                let system = "You are ASC Terminal. Speak in straight English. Recommend actions based on user intent. No hallucinations. If unsure, say 'Unknown'. Keep responses short and actionable."
                let response = try await runner.respond(system: system, user: command, history: history)
                history.append((command, response))
                lines.append(TerminalLine(prompt: "ASC RESPONSE", text: response))
            } catch {
                lines.append(TerminalLine(prompt: "ERROR", text: error.localizedDescription))
            }
            isLoading = false
        }
    }
}

struct ContentView: View {
    @StateObject private var vm = TerminalViewModel()
    @State private var showingImporter = false

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("ASC TERMINAL").font(.headline.monospaced()).foregroundStyle(.cyan)
                Spacer()
                Button("Import GGUF") { showingImporter = true }
                    .buttonStyle(.borderedProminent)
            }
            .padding()
            .background(Color.black)

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 10) {
                        ForEach(vm.lines) { line in
                            VStack(alignment: .leading, spacing: 3) {
                                if !line.prompt.isEmpty { Text(line.prompt).foregroundStyle(.cyan) }
                                Text(line.text).foregroundStyle(.green).textSelection(.enabled)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .id(line.id)
                        }
                        if vm.isLoading { ProgressView("Processing locally…").tint(.green) }
                    }
                    .padding()
                }
                .background(Color.black)
                .onChange(of: vm.lines.count) { _, _ in
                    if let id = vm.lines.last?.id { withAnimation { proxy.scrollTo(id, anchor: .bottom) } }
                }
            }

            HStack(alignment: .bottom) {
                Text("ASC >>").foregroundStyle(.cyan).font(.system(.body, design: .monospaced))
                TextField("command", text: $vm.input, axis: .vertical)
                    .textFieldStyle(.plain)
                    .foregroundStyle(.white)
                    .font(.system(.body, design: .monospaced))
                    .onSubmit { vm.submit() }
                Button("Run") { vm.submit() }.buttonStyle(.borderedProminent)
            }
            .padding()
            .background(Color(white: 0.08))

            Text(vm.modelName).font(.caption.monospaced()).foregroundStyle(.secondary).padding(.vertical, 5)
        }
        .preferredColorScheme(.dark)
        .fileImporter(isPresented: $showingImporter, allowedContentTypes: [UTType(filenameExtension: "gguf") ?? .data]) { result in
            if case .success(let url) = result { vm.importModel(url: url) }
        }
        .alert("ASC Terminal", isPresented: Binding(get: { vm.error != nil }, set: { if !$0 { vm.error = nil } })) {
            Button("OK") { vm.error = nil }
        } message: { Text(vm.error ?? "Unknown error") }
    }
}
