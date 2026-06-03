export default function Modal({
    setModal,
}: {
    setModal: React.Dispatch<React.SetStateAction<boolean>>;
}) {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
            <div className="text-center justify-center bg-white text-black rounded-3xl p-8">
                <h1 className="text-3xl font-bold mb-10">Advertencia</h1>
                <p>Recuerda que esta es una página de prueba, no debes tomar los resultados arrojados como si fueran 100% reales, debido a que el modelo no es exacto. Se recomienda ampliamente ir a un dentista.</p>
                <button className="bg-orange-500 px-6 py-2 rounded-3xl mt-4 " onClick={() => setModal(false)}>
                    Aceptar
                </button>
            </div>
        </div>
    );
}