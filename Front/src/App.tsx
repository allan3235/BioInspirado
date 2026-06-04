import { useEffect, useState } from 'react'
import './app.css'
import Modal from './components/Modal';
import type { ApiResponse } from './types/types';
import ModalResultado from './components/ModalResultado';



export default function App() {

  const [modal, setModal] = useState<boolean>(true);
  const [imagen, setImagen] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [data, setData] = useState<ApiResponse | null>(null);
  const [isSubmit, setIsSubmit] = useState<boolean>(false);
  const [modalRes, setModalRes] = useState<boolean>(false);


  const handleChange = (e: any) => {
    const file = e.target.files[0];
    setImagen(file);
    setPreview(URL.createObjectURL(file));
  }

  const handleEliminar = () => {
    setImagen(null);
    setPreview(null);
    setData(null);
    setIsSubmit(false);


  }

  async function handleSubmit(e: any) {
    e.preventDefault();
    if (!imagen) return;
    setData(null);
    setIsSubmit(true);
    const formData = new FormData();
    formData.append("file", imagen);

    try {
      const res = await fetch("http://localhost:8000/predecir/", {
        method: 'POST',
        body: formData,
      });

      const data: ApiResponse = await res.json();
      setData(data);
      setModalRes(true);
      console.log(data);

    } catch (error) {
      console.log(error);
    }




  }





  return (<>
    <div className="min-h-screen bg-white text-white">
      <main className="px-6 py-10 flex flex-col items-center gap-10">

        <form className="w-full max-w-5xl border-4 border-dashed border-blue-300 rounded-[40px] bg-blue-100 p-16 flex flex-col items-center justify-center gap-6 shadow-2xl">


          <div className="text-center">
            <p className="text-2xl font-bold text-black">
              Para empezar a analizar una imagen sube una imagen
            </p>
          </div>

          <label className="bg-blue-400 hover:bg-blue-600 transition-all px-6 py-2 rounded-lg border border-blue-400 cursor-pointer font-semibold shadow-md">
            Subir Imagen
            <input type="file" className="hidden" onChange={handleChange} />
          </label>
        </form>
        {preview && (
          <div className='w-full max-w-5xl flex justify-center flex-col items-center border-4 border-dashed border-blue-400 rounded-[40px] bg-blue-100 pb-10'>
            <h1 className='font-bold text-2xl text-black mb-4'>Imagen subida</h1>

            <img className='rounded-2xl mb-4' src={preview} alt='imagen' width={400} height={400} />
            <div className='flex flex-row gap-4'>
              <label className="bg-blue-400 hover:bg-blue-600 transition-all px-6 py-2 rounded-lg border border-blue-400 cursor-pointer font-semibold shadow-md">
                <button onClick={handleSubmit}>Analizar Imagen</button>
              </label>
              <label className="bg-blue-400 hover:bg-blue-600 transition-all px-6 py-2 rounded-lg border border-blue-400 cursor-pointer font-semibold shadow-md">
                <button onClick={handleEliminar}>Eliminar Imagen</button>
              </label>
            </div>

            {isSubmit && (<>
              {data ? (modalRes && (<ModalResultado data={data} setModal={setModalRes} />)) : (<div className="flex justify-center items-center">
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-gray-300 border-t-black"></div>
              </div>)}
            </>

            )}


          </div>
        )}
      </main>




    </div>
    {modal && (
      <Modal setModal={setModal} />
    )}

  </>
  )
}
