from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

# Definición del sistema de ecuaciones
def sistema(x, k1, k2, k3, F1, F2, F3):
    f1 = k1 * x[0] + np.sin(x[1]) - F1
    f2 = k2 * (x[1] - x[0]) + np.cos(x[2]) - F2
    f3 = k3 * x[2] - F3
    return np.array([f1, f2, f3])

# Método de Broyden
def metodo_broyden(f, x0, k1, k2, k3, F1, F2, F3, tol=1e-6, max_iter=100):
    n = len(x0)
    x = x0.copy()
    B = np.eye(n)
    Fx = f(x, k1, k2, k3, F1, F2, F3)
    
    iteraciones = [x.copy()]
    for i in range(max_iter):
        delta_x = np.linalg.solve(B, -Fx)
        x_new = x + delta_x
        Fx_new = f(x_new, k1, k2, k3, F1, F2, F3)
        iteraciones.append(x_new.copy())
        
        if np.linalg.norm(Fx_new) < tol:
            return x_new, iteraciones
        
        delta_F = Fx_new - Fx
        B = B + np.outer((delta_F - B @ delta_x), delta_x) / np.dot(delta_x, delta_x)
        
        x = x_new
        Fx = Fx_new
    
    return x, iteraciones

@app.route('/metodo_broyden', methods=['POST'])
def solve_broyden():
    data = request.json
    
    # Obtener parámetros del sistema del JSON
    k1 = data.get('k1', 1)
    k2 = data.get('k2', 1)
    k3 = data.get('k3', 1)
    F1 = data.get('F1', 0)
    F2 = data.get('F2', 0)
    F3 = data.get('F3', 0)
    x0 = data.get('x0', [0.1, 0.1, 0.1])
    
    try:
        x0 = np.array(x0, dtype=float)
        
        solucion, iteraciones = metodo_broyden(
            sistema, x0, k1, k2, k3, F1, F2, F3
        )
        
        # Preparar los datos para graficar
        iteraciones = np.array(iteraciones)
        iter_range = range(len(iteraciones))
        
        # Graficar la evolución de cada variable en las iteraciones
        plt.figure(figsize=(10, 6))
        plt.plot(iter_range, iteraciones[:, 0], label='x1')
        plt.plot(iter_range, iteraciones[:, 1], label='x2')
        plt.plot(iter_range, iteraciones[:, 2], label='x3')
        
        # Mostrar los valores finales en el gráfico
        plt.text(iter_range[-1], iteraciones[-1, 0], f'{iteraciones[-1, 0]:.4f}', fontsize=12, ha='left', color='blue')
        plt.text(iter_range[-1], iteraciones[-1, 1], f'{iteraciones[-1, 1]:.4f}', fontsize=12, ha='left', color='orange')
        plt.text(iter_range[-1], iteraciones[-1, 2], f'{iteraciones[-1, 2]:.4f}', fontsize=12, ha='left', color='green')
        
        plt.xlabel('Número de iteraciones')
        plt.ylabel('Valor de las variables')
        plt.title('Evolución de las soluciones en el método de Broyden')
        plt.legend()
        plt.grid(True)
        
        # Convertir la gráfica a base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
        
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        
        return jsonify({'Solucion': solucion.tolist(), 'Iteraciones': iteraciones.tolist(), 'Imagen': img_base64})
    
    except Exception as e:
        return jsonify({'error': f'Error durante la ejecución: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)