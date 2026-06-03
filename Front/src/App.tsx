import { useEffect, useState } from 'react'
import './app.css'
import Modal from './components/Modal';

type ApiResponse ={
  filename:string;
  content_type:string;
  message:string;
}

export default function App() {

  const [modal, setModal] = useState<boolean>(true);
  const [imagen, setImagen] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [data,setData] = useState<ApiResponse | null>(null);
  

  const handleChange = (e: any) => {
    const file = e.target.files[0];
    setImagen(file);
    setPreview(URL.createObjectURL(file));
  }

  async function handleSubmit(e: any) {
    e.preventDefault();
    if (!imagen) return;
    const formData = new FormData();
    formData.append("file", imagen);
    try {
      const res = await fetch("http://localhost:8000/predecir/", {
        method: 'POST',
        body: formData,
      });

      const data : ApiResponse = await res.json();
      setData(data);
      console.log(data);

    }catch(error){
      console.log(error);
    }

    


  }





  return (<>
    <div className="min-h-screen bg-black text-white">
      <main className="px-6 py-10 flex flex-col items-center gap-10">

        <form className="w-full max-w-5xl border-4 border-dashed border-purple-300 rounded-[40px] bg-purple-950 p-16 flex flex-col items-center justify-center gap-6 shadow-2xl">


          <div className="text-center">
            <p className="text-2xl font-bold">
              Para empezar a analizar una imagen sube una
            </p>

            <p className="text-purple-200 mt-2">
              Imágenes tipo JPG o PNG
            </p>
          </div>

          <label className="bg-purple-700 hover:bg-purple-600 transition-all px-6 py-2 rounded-lg border border-purple-300 cursor-pointer font-semibold shadow-md">
            Subir Imagen
            <input type="file" className="hidden" onChange={handleChange} />
          </label>
        </form>
        {preview && (
          <div className='w-full max-w-5xl flex justify-center flex-col items-center border-4 border-dashed border-purple-300 rounded-[40px] bg-purple-950 pb-10'>
            <h1>Imagen subida</h1>

            <img className='rounded-2xl' src={preview} alt='imagen' width={200} height={200} />
            <button onClick={handleSubmit}>Analizar</button>
            {data &&(<div>
              <p>{data.filename}</p>
              <p>{data.content_type}</p>
              <p>{data.message}</p>
            </div>)}
            
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
