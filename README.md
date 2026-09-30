# ASCTerminal — iPhone

Native SwiftUI port of the original ASC Terminal concept: an ASCII-infused, lightweight local AI terminal that gives short, actionable responses from a local GGUF model.

## iPhone use
1. Build the project with the included GitHub Action on macOS.
2. Put a compatible `.gguf` model in Files/iCloud Drive.
3. Open ASC Terminal and tap **Import GGUF**.
4. Type commands at `ASC >>`. Inference runs locally through llama.cpp.

The original Python implementation used Phi-3-mini-4k-instruct. The native app accepts GGUF models so the model is not incorrectly hard-coded or bundled into a huge iOS artifact.

## Build
The workflow builds the official llama.cpp XCFramework, generates the Xcode project with XcodeGen, builds the iOS Simulator app unsigned, and uploads a clean `.app` plus source ZIP.
