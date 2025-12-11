# MONITOREO Y MANTENIMIENTO DEL MODELO DE CHURN

Este documento describe cómo se monitorea y mantiene en el tiempo el modelo de
predicción de no recompra (churn a 30 días) del caso **Emporio Vinos y Licores**.

---

## 1. Alcance del monitoreo

El monitoreo considera dos niveles:

1. **Monitoreo técnico del servicio**
   - Disponibilidad de la API.
   - Latencia de las respuestas.
   - Errores de la API.

2. **Monitoreo del rendimiento del modelo**
   - Calidad de las predicciones (métricas de clasificación).
   - Posibles cambios en los datos (*data drift*).

El objetivo es detectar de forma temprana cualquier degradación del servicio o
del modelo, para poder aplicar reentrenamiento o rollback si es necesario.

---

## 2. Métricas técnicas

**Fuente:** API FastAPI desplegada en Docker.

### 2.1 Endpoints de referencia

- `GET /health`  
  Responde con el estado del modelo y el número de features esperadas.  
  Se usa como *smoke test* después del despliegue y para chequeos periódicos.

- `POST /predict`  
  Endpoint principal de inferencia. Toda llamada queda registrada a nivel de logs
  del contenedor Docker.

### 2.2 Métricas técnicas a monitorear

- **Disponibilidad:** porcentaje de tiempo en que `/health` responde con `status = "ok"`.
- **Latencia:** tiempo promedio de respuesta de `/predict` (p. ej., p95 < 300 ms).
- **Tasa de errores:** porcentaje de requests con código HTTP 5xx (errores internos)
  o 4xx inesperados.

En un entorno productivo real, estas métricas se podrían recolectar con herramientas
como Prometheus, Grafana o servicios de monitoreo en la nube. En este caso, el
monitoreo se realiza a partir de:

- Logs del contenedor (`docker logs emporio-churn-api`).
- Pruebas periódicas manuales o automatizadas contra `/health` y `/predict`.

---

## 3. Métricas de negocio y calidad del modelo

**Objetivo del modelo:** identificar clientes con alta probabilidad de **no recompra
en 30 días** para priorizar acciones comerciales.

### 3.1 Métricas principales

- **PR-AUC (Área bajo curva Precisión–Recall):** métrica usada para seleccionar el
  mejor modelo durante el entrenamiento.
- **ROC-AUC:** medida global de discriminación del modelo.
- **Precisión y recall en el Top 10 % de clientes con mayor riesgo**  
  (según el umbral `threshold_top10` aprendido en entrenamiento).

En el entorno actual, estas métricas se calculan en `src/train_model.py` y se
guardan como parte del proceso de entrenamiento. Para un monitoreo continuo
en producción se requiere:

- Comparar periódicamente las predicciones del modelo con los datos reales de
  recompra/no recompra en ventanas de tiempo recientes.
- Recalcular métricas como precisión, recall y PR-AUC sobre datos nuevos.

---

## 4. Data drift y degradación del modelo

Con el paso del tiempo, el comportamiento de los clientes puede cambiar. Esto se
refleja en:

- Distribución distinta de variables como `freq_180`, `spend_total`, `recency_days`,
  canales de compra o regiones.
- Aparición de nuevos patrones de compra que el modelo original no vio.

Se consideran señales de posible *drift* o degradación cuando:

- La distribución de features actuales se aleja significativamente de la distribución
  observada en el dataset de entrenamiento (por ejemplo, variaciones fuertes en medias
  y desviaciones estándar).
- Las métricas de negocio (precisión y recall en el Top 10 % de riesgo) caen por debajo
  de un umbral definido por el negocio.

En esos casos, se debe evaluar un **reentrenamiento** del modelo.

---

## 5. Plan de reentrenamiento

El modelo se entrena en `src/train_model.py` y guarda un artefacto en
`models/modelo_churn_rf_vX.joblib`.

### 5.1 Cuándo reentrenar

Se recomienda reentrenar cuando se cumpla alguna de estas condiciones:

- Cada **3 a 6 meses**, como política preventiva.
- Cuando se detecte una degradación relevante de las métricas (por ejemplo,
  caída sostenida de más de 10–15 % en PR-AUC o en recall del Top 10 %).
- Cuando cambie la lógica de negocio (nuevos productos, nuevos canales de venta,
  cambios importantes en campañas comerciales).

### 5.2 Procedimiento de reentrenamiento

1. Extraer datos actualizados de usuarios y órdenes.
2. Actualizar el script de entrenamiento si es necesario.
3. Ejecutar:

   ```bash
   .venv\Scripts\activate
   python src/train_model.py
   pytest



---

Con esto ya tienes un **documento de monitoreo y mantenimiento** súper defendible para la rúbrica y para la defensa oral.  
Si quieres, después armamos uno parecido para el **pipeline/CI-CD** o un resumen de todo para el informe.
