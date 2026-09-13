# Entrega de modelos desde Colab

Cada responsable entrega un **bundle**, no sólo un pickle. Mantener onboarding, transacción y secuencia en paquetes independientes.

## Archivos obligatorios

1. Notebook ejecutado de principio a fin, sin celdas ambiguas.
2. Modelo exportado y todo scaler, encoder, selector o calibrador ajustado.
3. `requirements.txt` con versiones exactas y versión de Python del runtime.
4. Dataset: nombre, versión/URL, SHA-256 del archivo, target y columnas excluidas.
5. Split: semilla, estrategia y lista/archivo con identificadores de train, validation y test.
6. Features crudas y finales en orden, con tipo, nulabilidad, categorías y tratamiento de desconocidos.
7. Dos o más casos dorados producidos en Colab: payload crudo, vector final y score esperado.

Ejemplo mínimo de caso dorado:

```json
{
  "input": {
    "entity_id": "golden-001",
    "as_of": "2026-09-13T09:00:00Z",
    "application": {},
    "events": []
  },
  "expected": {
    "model_vector": [[0.12, -0.44]],
    "risk_score": 0.7312
  },
  "atol": 1e-7
}
```

Las expectativas se calculan en el notebook original, nunca llamando al runner que se intenta validar.

## Exportación recomendada

Para sklearn/XGBoost/LightGBM, exportar un `Pipeline` completo cuando sea posible. Si se entrega un diccionario, usar claves `model`, `preprocessor`, `selector` opcional y `feature_names`. Para Keras, usar `.keras` y exportar el scaler por separado. Registrar el método de salida: `predict_proba`, `predict`, `decision_function` o `reconstruction_mse`.

El LSTM-AE debe incluir además forma del tensor, longitud, mínimo de historia, orden temporal, padding, máscara, valor de padding, función exacta de error y distribución/umbral de validación usada para normalizarlo.

Antes de compartir, reiniciar el runtime, ejecutar todo y comprobar que el artefacto recién guardado reproduce los casos dorados. No volver a ajustar transformadores durante serving.
