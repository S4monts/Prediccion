from wtforms import ValidationError
import pandas as pd
import numpy as np

def validate_batch_file(file):
    try:
        if file.filename == '':
            return False, "No se seleccionó ningún archivo"
            
        if not (file.filename.endswith('.csv') or file.filename.endswith(('.xlsx', '.xls'))):
            return False, "Formato no soportado. Use archivos CSV o Excel"
        
        # Leer el archivo
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        
        # Validar estructura
        required_columns = 30
        if len(df.columns) != required_columns:
            return False, f"El archivo debe tener exactamente {required_columns} columnas. Se encontraron {len(df.columns)}"
        
        # Validar nombres de columnas (opcional)
        expected_cols = [f'C{i}' for i in range(1, 31)]
        if not all(col in df.columns for col in expected_cols):
            return False, f"Las columnas deben llamarse: {', '.join(expected_cols)}"
        
        # Validar valores numéricos
        if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
            return False, "Todas las columnas deben contener valores numéricos"
            
        return True, df
        
    except Exception as e:
        return False, f"Error al procesar el archivo: {str(e)}"