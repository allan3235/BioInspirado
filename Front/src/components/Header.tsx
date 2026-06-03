export default function Header() {
    return (
        <header className="bg-purple-800 py-5 px-8 flex items-center justify-center gap-6 shadow-lg">
            <div className="text-5xl">🦷</div>

            <h1 className="text-2xl md:text-4xl font-bold text-center uppercase tracking-wide leading-tight">
                Detección de Caries en Radiografías mediante
                <br />
                Redes Neuronales Convolucionales y Binarizadas
            </h1>
        </header>
        )
}