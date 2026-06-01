# Novel-TTS-Frontend 🎙️📖

A powerful, visual React + Vite frontend workstation designed to convert novels and long-form text into high-quality, multi-character audiobooks using Google's Gemini TTS API. 

This project provides an interactive interface for script tagging, role assignment, smart text chunking, and batch text-to-speech generation.

## 🌟 Features

- **📖 Novel Script Processing**: Automatically parse and chunk long novels while preserving semantic boundaries (sentences and paragraphs).
- **🎭 Multi-Character Cast Management**: Assign different voices, aliases, and specific TTS models to different characters and narrators.
- **🏷️ Smart Dialogue Tagging**: Built-in prompts to structure plain text into JSON-driven scripts, identifying spoken dialogue vs. narration.
- **✂️ Intelligent Text Chunking**: Proportional slicing algorithm to break down massive texts without breaking quotes or sentences.
- **🔊 Visual Audio Workspace**: Monitor TTS generation status in real-time with an interactive canvas for generated audio blocks.
- **💾 Local First (IndexedDB)**: Automatically saves your workspace state, scripts, and audio blobs locally in the browser so you never lose progress.

## 🚀 Tech Stack

- **Framework**: [React 19](https://react.dev/) + [Vite](https://vitejs.dev/)
- **Styling**: [Tailwind CSS](https://tailwindcss.com/) v4
- **Icons**: [Lucide React](https://lucide.dev/)
- **Storage**: IndexedDB for local persistent storage

## 🛠️ Getting Started

### Prerequisites
Make sure you have [Node.js](https://nodejs.org/) installed on your machine.

### Installation

1. Clone the repository:
```bash
git clone https://github.com/wpuu/Novel-TTS-Frontend.git
cd Novel-TTS-Frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

4. Open your browser and navigate to the provided local URL (usually `http://localhost:5173`).

### Build for Production

To build the app for production, run:
```bash
npm run build
```
The optimized files will be generated in the `dist` directory.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/wpuu/Novel-TTS-Frontend/issues).

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
