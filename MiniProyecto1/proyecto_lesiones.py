import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

class Preprocesamiento:
    def __init__(self, ruta_datos, usar_muestra=False, columnas=None):
        self.ruta_datos = ruta_datos
        self.usar_muestra = usar_muestra
        self.columnas = columnas
        self.data = None

    def cargar_datos(self):
        # Carga los datos desde el archivo especificado.
        self.data = pd.read_csv(self.ruta_datos)
        if self.columnas:
            self.data = self.data[self.columnas]
        if self.usar_muestra:
            self.data = self.data.sample(frac=0.1, random_state=42)
        print("Datos cargados correctamente.")

    def diagnostico_datos(self, descriptores, estadisticas=None):
        # Genera un diagnóstico descriptivo del dataset.
        if estadisticas is None:
            estadisticas = ['mean', 'std', 'min', 'max']
        diagnostico = {}
        for estadistica in estadisticas:
            if estadistica == 'mean':
                diagnostico['media'] = self.data[descriptores].mean()
            elif estadistica == 'std':
                diagnostico['desviacion_estandar'] = self.data[descriptores].std()
            elif estadistica == 'min':
                diagnostico['valor_minimo'] = self.data[descriptores].min()
            elif estadistica == 'max':
                diagnostico['valor_maximo'] = self.data[descriptores].max()
        diagnostico['valores_faltantes'] = self.data[descriptores].isnull().sum()
        return diagnostico

    def imputar_valores(self, descriptores, estrategias):
        # Imputa valores faltantes según las estrategias proporcionadas.
        for col, estrategia in estrategias.items():
            if estrategia == 'mean':
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            elif estrategia == 'median':
                self.data[col].fillna(self.data[col].median(), inplace=True)
            elif estrategia == 'mode':
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        print("Valores imputados correctamente.")

    def generar_graficos(self, columnas, tipo='scatter', guardar=False):
        # Genera gráficos exploratorios.
        col_x, col_y = columnas
        plt.figure(figsize=(8, 6))
        if tipo == 'scatter':
            sns.scatterplot(data=self.data, x=col_x, y=col_y)
        elif tipo == 'box':
            sns.boxplot(data=self.data, x=col_x, y=col_y)
        else:
            raise ValueError("Tipo de gráfico no válido. Use 'scatter' o 'box'.")
        plt.title(f'{tipo.capitalize()} plot de {col_x} y {col_y}')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        if guardar:
            plt.savefig(f'graficos/{tipo}_plot_{col_x}_vs_{col_y}.png')
        else:
            plt.show()

    def escalar_datos(self, descriptores, estrategias):
        # Escala o normaliza los datos según las estrategias proporcionadas.
        for col, estrategia in estrategias.items():
            if estrategia == 'z-score':
                scaler = StandardScaler()
            elif estrategia == 'min-max':
                scaler = MinMaxScaler()
            else:
                raise ValueError("Estrategia no válida. Use 'z-score' o 'min-max'.")
            self.data[col] = scaler.fit_transform(self.data[[col]])
        print("Datos escalados correctamente.")

    def dividir_datos(self, columna_objetivo, test_size=0.2, random_state=42):
        # Divide los datos en conjuntos de entrenamiento y prueba.
        X = self.data.drop(columns=[columna_objetivo])
        y = self.data[columna_objetivo]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return X_train, X_test, y_train, y_test

    def ejecutar_procesamiento(self, descriptores, estrategias_imputacion, estrategias_escalamiento):
        # Aplica el flujo completo de preprocesamiento.
        self.cargar_datos()
        print("Diagnóstico inicial:")
        print(self.diagnostico_datos(descriptores))
        self.imputar_valores(descriptores, estrategias_imputacion)
        self.escalar_datos(descriptores, estrategias_escalamiento)


def clasificador(tipo, ruta_guardado, X_train, y_train, **kwargs):
    # Entrena y guarda un modelo clasificador.
    if tipo == 'naive_bayes':
        model = GaussianNB(**kwargs)
    elif tipo == 'logistic_regression':
        model = LogisticRegression(**kwargs, max_iter=1000, solver='liblinear')
    else:
        raise ValueError("Tipo no válido. Use 'naive_bayes' o 'logistic_regression'.")
    model.fit(X_train, y_train)
    with open(ruta_guardado, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo {tipo} guardado en {ruta_guardado}.")

def evaluar_rendimiento(ruta_modelo, X_test, y_test, tipo_analisis='metricas'):
    # Evalúa el modelo en datos de prueba.
    with open(ruta_modelo, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    if tipo_analisis == 'metricas':
        print("Reporte de Clasificación:")
        print(classification_report(y_test, y_pred, zero_division=1))
    elif tipo_analisis == 'matriz_confusion':
        print("Matriz de Confusión:")
        print(confusion_matrix(y_test, y_pred, zero_division=1))
    else:
        raise ValueError("Tipo de análisis no válido. Use 'metricas' o 'matriz_confusion'.")
        