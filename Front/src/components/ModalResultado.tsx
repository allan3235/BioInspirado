import type { ApiResponse } from "../types/types";

export default function ModalResultado({ data, setModal }: { data: ApiResponse, setModal: React.Dispatch<React.SetStateAction<boolean>> }) {
    const newData = data.probabilidades.filter((nombre) => nombre.nombre !== data.diagnostico);
    return (
        <div className="fixed inset-0 flex items-center justify-center bg-black/50">
            <div className="flex flex-col text-center justify-center bg-white text-black rounded-3xl p-8 gap-3">
                <h1 className="text-3xl">Resultados</h1>
                <p>El resultado del modelo indica que la enfermedad dental más probable es <span className="font-bold">{data.diagnostico}</span> con un <span className="font-bold">{data.confianza}% </span>de confianza</p>
                <hr />
                <h1 className="mb-5 text-3xl">Probabilidad de otras enfermedades</h1>
                <div className="flex flex-row gap-3 justify-center">
                    {newData.map((prob) => (
                        <>
                            <p>{prob.nombre}: <span className="font-bold">{prob.probabilidad}%</span></p>
                        </>
                    ))}
                </div>
                <label className="bg-blue-400 hover:bg-blue-600 transition-all px-6 py-2 rounded-lg border border-blue-400 cursor-pointer font-semibold shadow-md">
                    <button onClick={() => setModal(false)}>Aceptar</button>
                </label>



            </div>
        </div>
    );
}