export default function App() {
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Header */}
      <header className="bg-purple-800 py-5 px-8 flex items-center justify-center gap-6 shadow-lg">
        <div className="text-5xl">🦷</div>

        <h1 className="text-2xl md:text-4xl font-bold text-center uppercase tracking-wide leading-tight">
          Detección de Caries en Radiografías mediante
          <br />
          Redes Neuronales Convolucionales y Binarizadas
        </h1>
      </header>

      {/* Main */}
      <main className="px-6 py-10 flex flex-col items-center gap-10">
        {/* Upload button */}
        <div className="w-full flex justify-end max-w-6xl">
          <label className="bg-purple-700 hover:bg-purple-600 transition-all px-6 py-2 rounded-lg border border-purple-300 cursor-pointer font-semibold shadow-md">
            Subir Imagen
            <input type="file" className="hidden" />
          </label>
        </div>

        {/* Upload Box */}
        <div className="w-full max-w-5xl border-4 border-dashed border-purple-300 rounded-[40px] bg-purple-950 p-16 flex flex-col items-center justify-center gap-6 shadow-2xl">
          <div className="text-8xl">☁️</div>

          <div className="text-center">
            <p className="text-2xl font-bold">
              Suba y pegue su imagen aquí
            </p>

            <p className="text-purple-200 mt-2">
              Imágenes tipo JPG o PNG
            </p>
          </div>

          <button className="bg-purple-700 hover:bg-purple-600 transition-all px-10 py-3 rounded-xl border border-purple-300 font-semibold text-lg shadow-lg">
            Analizar Imagen
          </button>
        </div>

        {/* Results Section */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full max-w-5xl">
          {/* Image preview */}
          <div className="bg-zinc-900 border border-zinc-700 rounded-2xl p-4 shadow-xl flex items-center justify-center">
            <img
              src="https://upload.wikimedia.org/wikipedia/commons/0/05/Dental_radiograph.png"
              alt="Radiografía"
              className="rounded-xl object-cover max-h-[400px]"
            />
          </div>

          {/* Result card */}
          <div className="bg-zinc-100 text-black rounded-2xl p-8 shadow-xl flex flex-col items-center justify-center gap-6">
            <h2 className="text-2xl font-bold">
              Análisis de Resultados
            </h2>

            {/* Fake Gauge */}
            <div className="relative w-64 h-32 overflow-hidden">
              <div className="absolute bottom-0 w-64 h-64 border-[30px] border-purple-700 rounded-full"></div>

              <div className="absolute bottom-2 left-1/2 w-2 h-28 bg-black origin-bottom rotate-45"></div>
            </div>

            <div className="text-center space-y-2">
              <p className="text-lg">
                Detección de Caries:{" "}
                <span className="text-red-600 font-bold">
                  Presencia de Caries
                </span>
              </p>

              <p className="text-xl font-bold">
                Probabilidad: 47%
              </p>
            </div>

            <div className="w-full space-y-4 text-sm">
              <div>
                <p className="font-semibold">Detalles de Caries</p>
                <div className="h-[2px] bg-zinc-400 mt-2"></div>
              </div>

              <div>
                <p className="font-semibold">Detalles de la Probabilidad</p>
                <div className="h-[2px] bg-zinc-400 mt-2"></div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="bg-purple-950 py-6 text-center text-purple-200 mt-10 border-t border-purple-800">
        Sistema de detección de caries mediante IA
      </footer>
    </div>
  )
}
