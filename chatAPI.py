from fastapi import FastAPI
from pydantic import BaseModel
from typing import Text
import ast
import os
import pandas as pd
import numpy as np
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()
path_file1 = "iso9001.csv"
datos_ISO = pd.read_csv(path_file1)

path_file2 = "FDSA.csv"
datos_SERV = pd.read_csv(path_file2)
client = OpenAI(
    organization= os.getenv('ORGANIZATION'),
    api_key= os.getenv('API_KEY')
)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def buscar(busqueda, datos, n_resultados=5):
    busqueda_embed = get_embedding(busqueda, model="text-embedding-3-small")
    datos["Similitud"] = datos['Embedding'].apply(lambda x: cosine_similarity(ast.literal_eval(x), busqueda_embed))
    datos = datos.sort_values("Similitud", ascending=False)
    return datos.iloc[:n_resultados][["texto", "Similitud", "Embedding"]]
app = FastAPI()

msj = []
class Msj(BaseModel):
    type: str
    mensaje: Text

msj = Msj
@app.get("/")
def inicio():
    return {"message": "Esta es una Api para realizar consultas a OpenAI"}


@app.post("/consulta")
def procesar(msj:Msj):
    mens = msj.dict()
    if mens["type"] == "ISO":
        prompt = mens["mensaje"]
        dat = buscar(prompt, datos_ISO, 2)
        embed = dat['texto'].str.cat()  # aqui estan concatenados tos los embeddings

        query = f""" Usa la siguiente información para contestar la pregunta, si no encuentras la 
        respuesta, contesta amablemente diciendo que en el momento no dispones de la información solicitada, en caso de que 
        la pregunta sea un saludo debes contestar amablemente al saludo indicando tu nombre y el de tu creador"
        información:
        \"\"\"
        {embed}
        \"\"\"
        pregunta:
        \"\"\"
        {prompt}
        \"\"\" 
        """
        rol = "Eres EMMA creada por Ingenieros de InproExt, un asistente experto en la norma iso 9001. Tu objetivo es proporcionar respuestas claras y precisas, con un límite máximo de 70 palabras"
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": rol},
                {"role": "user", "content": query}],
            max_tokens=400,
            temperature=0.8,
            #stream=True,
        )
        print(stream.choices[0].message.content)

        return [{"type":"ISO","mensaje":stream.choices[0].message.content}]
    if mens["type"] == "SERVICIO":
        prompt = mens["mensaje"]
        dat = buscar(prompt, datos_SERV, 2)
        embed = dat['texto'].str.cat()  # aqui estan concatenados tos los embeddings
        query = f""" Usa la siguiente información para contestar la pregunta, si no encuentras la 
        respuesta, contesta amablemente diciendo que en el momento no dispones de la información solicitada, en caso de que 
        la pregunta sea un saludo debes contestar amablemente al saludo indicando tu nombre y el de tu creador"
        información:
        \"\"\"
        {embed}
        \"\"\"
        pregunta:
        \"\"\"
        {prompt}
        \"\"\" 
        """
        rol = "Eres EMMA, creada por Ingenieros de InproExt, un asistente experto en servicio al cliente. Tu objetivo es proporcionar respuestas claras y precisas, con un límite máximo de 50 palabras"
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": rol},
                {"role": "user", "content": query}],
            max_tokens=400,
            temperature=0.8,
            #stream=True,
        )
        print(stream.choices[0].message.content)

        return [{"type": "SERVICIO", "mensaje": stream.choices[0].message.content}]
