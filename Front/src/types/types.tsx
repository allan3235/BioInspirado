export type ApiResponse = {
    filename: string;
    diagnostico: string;
    confianza: number;
    probabilidades: Probabilidades[]
}
type Probabilidades = {
    nombre: string;
    probabilidad: number;
}