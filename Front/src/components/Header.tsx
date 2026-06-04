export default function Header() {
    return (
        <header className="bg-blue-300 py-5 px-8 flex items-center justify-center gap-6 shadow-lg">
            <div className="text-5xl">🦷</div>

            <h1 className="text-2xl md:text-4xl font-bold text-center uppercase tracking-wide leading-tight">
                Detección de enfermedades bucales mediante imágenes utilizando
                <br />
                Redes Neuronales Convolucionales
            </h1>
        </header>
        )
}