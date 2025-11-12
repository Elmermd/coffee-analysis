"""
Coffee Survey Data Cleaning and Transformation Utilities 
=============================================================

Funciones helper para limpieza, transformación y preparación
del dataset de encuesta de café

Autor: Elias Basaldua Garcia
Fecha: Noviembre 2025
"""

import pandas as pd 
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------------
# DICCIONARIOS DE MAPEO PARA VARIABLES ORDINALES
# -------------------------------------------------------------

AGE_ORDER = {
  '<18 years old': 0,
  '18-24 years old': 1,
  '25-34 years old': 2,
  '35-44 years old': 3,
  '45-54 years old': 4,
  '55-64 years old': 5,
  '>65 years old': 6
}

AGE_LABELS = {
  0: '<18',
  1: '18-24',
  2: '25-34',
  3: '35-44',
  4: '45-54',
  5: '55-64',
  6: '>65'
}

CUPS_ORDER = {
  'Less than 1': 0,
  '1': 1,
  '2': 2,
  '3': 3,
  '4': 4,
  'More than 4': 5
}

EDUCATION_ORDER = {
  'Less than high school': 0,
  'High school graduate': 1,
  "Some college or associate's degree": 2,
  "Bachelor's degree": 3,
  "Master's degree": 4,
  "Doctorate or professional degree": 5
}

EMPLOYMENT_ORDER = {
  'Unemployed': 0,
  'Student': 1,
  'Employed part-time': 2,
  'Employed full-time': 3,
  'Retired': 4
}

CHILDREN_ORDER = {
  '0': 0,
  '1': 1,
  '2': 2,
  '3': 3,
  'More than 3': 4
}

# -------------------------------------------------------------
# FUNCIONES DE LIMPIEZA GENERAL
# -------------------------------------------------------------

def load_and_initial_clean(filepath: str) -> pd.DataFrame:
  """
  Carga el dataset y realiza limpieza inicial básica

  Parameters:
  -----------
  filepath : str
    Ruta al archivo CSV
  
  Returns:
  --------
  pd.DataFrame
    Dataset Limpio
  """

  # Cargar datos
  df = pd.read_csv(filepath)

  # Eliminar BOM si existe
  df.columns = df.columns.str.replace('\ufeff', '')

  # Renombrar columnas para facilitar manejo
  df.columns = df.columns.str.strip()

  print(f"Dataset cargado: {df.shape[0]} filas | {df.shape[1]} columnas")

  return df

def remove_high_missing_columns(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
  """
  Elimina columnas con porcentaje de missing superior al threshold.

  Parameters:
  -----------
  df : pd.DataFrame
    Dataset original
  threshold : float
    Umbral de missing (default: 0.95 = 95%)
  
  Returns:
  --------
  pd.DataFrame
    Dataset sin columnas de alta missing
  """

  missing_pct = df.isnull().sum() / len(df)
  cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()

  print(f"Eliminando {len(cols_to_drop)} columnas con >{threshold*100}% missing")
  if cols_to_drop:
    print(f"  Columnas eliminadas: {cols_to_drop[:5]}...")

  return df.drop(columns=cols_to_drop)

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
  """
  Estandariza nombres de columnas: minúsculas, sin espacios, sin caracteres especiales

  Parameters:
  -----------
  df.pd.DataFrame
    Dataset original
  
  Returns:
  --------
  pd.DataFrame
    Dataset con nombres de columnas estandarizadas
  """

  # Crear diccionario de mapeo para columnas demográficas clave
  rename_dict = {
    'Submission ID': 'submission_id',
    'What is your age?': 'age',
    'How many cups of coffee do you typically drink per day?': 'cups_per_day',
    'Gender': 'gender',
    'Education Level': 'education',
    'Ethnicity/Race': 'ethnicity',
    'Employment Status': 'employment',
    'Number of Children': 'children',
    'Political Affiliation': 'political_affiliation'
  }

  df = df.rename(columns=rename_dict)
  print("Columnas demográficas estandarizadas")

  return df

# -------------------------------------------------------------
# TRANSFORMACIÓN DE VARIABLES CATEGÓRICAS
# -------------------------------------------------------------

def encode_ordinal_variable(
    df:pd.DataFrame,
    column: str,
    mapping: Dict,
    new_column_suffix: str = '_encoded'
) -> pd.DataFrame:
  """
  Codifica una variable ordinal categórica a numérica

  Parameters:
  -----------
  df : pd.DataFrame
    Dataset
  column : str
    Nombre de la columna a codificar
  mapping : Dict
    Diccionario de mapeo categoría a número
  new_column_suffix : str
    Sufijo para nueva columna (default: '_encoded')
  
  Returns:
  --------
  pd.DataFrame
    Dataset con nueva columna codificada
  """

  new_col = column + new_column_suffix
  df[new_col] = df[column].map(mapping) # La función .map convierte texto a número usando un diccionario

  # Reportar valores no mapeados
  unmapped = df[df[new_col].isna() & df[column].notna()][column].unique()
  """
  df[new_col].isna(): 
  Devuelve True donde la nueva columna (####_encoded) tiene un valor faltante (NaN). 
  Es decir, donde el mapeo no encontró correspondencia.

  df[column].notna():
  Devuelve True donde la columna original (Nivel) no está vacía (o sea, no era un NaN ya desde antes).

  & (AND lógico):
  Solo queremos las filas donde
  - La nueva columna es NaN (falló el mapeo),
  - y la columna original sí tenía algo (no estaba vacía).
  """
  if len(unmapped) > 0:
    print(f"Valores no mapeados en '{column}': {unmapped}")
  
  return df

def encode_all_ordinals(df: pd.DataFrame) -> pd.DataFrame:
  """
  Codifica todas las variables ordinales principales

  Parameters:
  -----------
  df: pd.DataFrame
    Dataset con columnas estandarizadas
  
  Returns:
  --------
  pd.DataFrame
    Dataset con variables ordinales codificadas
  """
  print("Codificando variables ordinales")

  # Edad
  if 'age' in df.columns:
    df = encode_ordinal_variable(df,'age',AGE_ORDER)
    print(" age -> age_encoded")

  # Tazas por día
  if 'cups_per_day' in df.columns:
    df = encode_ordinal_variable(df, 'cups_per_day', CUPS_ORDER)
    print(" cups_per_day -> cups_per_day_encoded") 

  # Educación
  if 'education' in df.columns:
    df = encode_ordinal_variable(df, 'education',EDUCATION_ORDER)
    print(" education -> education_encoded")
  
  # Empleo
  if 'employment' in df.columns:
    df = encode_ordinal_variable(df,'employment',EMPLOYMENT_ORDER)
    print(" employment -> employment_encoded")
  
  # Hijos (si existe como string)
  if 'children' in df.columns and df['children'].dtype == 'object':
    df = encode_ordinal_variable(df,'children',CHILDREN_ORDER)
    print(" children -> children_encoded")
  
  return df

def create_consumption_segment(df:pd.DataFrame, cups_col: str = 'cups_per_day_encoded') -> pd.DataFrame:
  """
  Crea segmentos de consumo basados en tazas por día

  Parameters:
  -----------
  df: pd.DataFrame
    Dataset con columna de tazas codificada
  cups_col : str
    Nombre de la columna de tazas codificada
  
  Returns:
  --------
  pd.DataFrame
    DataFrame con nueva columna 'consumption_segment'
  """
  def segment_consumer(cups):
    if pd.isna(cups):
      return 'Unkown'
    elif cups == 0:
      return 'Light (<1 cup)'
    elif cups in [1,2]:
      return 'Moderate (1-2 cups)'
    else:
      return 'Heavy (3+ cups)'
  
  df['consumption_segment'] = df[cups_col].apply(segment_consumer)
  print("Segmentos de consumo creados: Light, Moderate, Heavy")
  return df

def create_age_groups(df: pd.DataFrame, age_col: str = 'age_encoded') -> pd.DataFrame:
  """
  Crea grupos de edad más amplios para análisis.

  Parameters:
  -----------
  df : pd.DataFrame
    Dataset con columna de edad codificada
  age_Col : str
    Nombre de la columna de edad codificada

  Returns:
  --------
  pd.DataFrame
    Dataset con nueva columna 'age_group' 
  """
  def group_age(age_code):
    if pd.isna(age_code):
      return 'Unknown'
    if age_code <= 1: # <18, 18-24
      return 'Gen Z (<25)'
    if age_code == 2: # 25-34
      return 'Young Millennials (25-34)'
    if age_code == 3: # 35-44
      return 'Older Millennials (35-44)'
    if age_code in [4,5]: # 45-54, 55-64
      return 'Gen X (45-64)'
    else: # 65+
      return 'Boomers+ (65+)'
    
  df['age_group'] = df[age_col].apply(group_age)

  print("Grupos de edad creados: Gen Z, Millenials, Gen z, Boomers+")

  return df

# -------------------------------------------------------------
# MANEJO DE VALORES FALTANTES
# -------------------------------------------------------------

def impute_demographic_missing(df: pd.DataFrame, strategy: str = 'mode') -> pd.DataFrame:
    """
    Imputa valores faltantes en variables demográficas.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    strategy : str
        Estrategia de imputación ('mode' o 'unknown')
        
    Returns:
    --------
    pd.DataFrame
        Dataset con valores imputados
    """
    demographic_cols = ['gender', 'education', 'employment', 'political_affiliation']
    
    for col in demographic_cols:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                if strategy == 'mode':
                    mode_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
                    print(f"  {col}: {missing_count} valores imputados con moda ({mode_value})")
                else:
                    df[col].fillna('Unknown', inplace=True)
                    print(f"  {col}: {missing_count} valores imputados con 'Unknown'")
    
    return df

def fill_binary_columns_with_false(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena valores NaN en columnas binarias (True/False) con False.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
        
    Returns:
    --------
    pd.DataFrame
        Dataset con columnas binarias completas
    """
    # Identificar columnas binarias
    binary_cols = []
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2 and set(unique_vals).issubset({True, False, 'True', 'False'}):
            binary_cols.append(col)
    
    # Rellenar con False
    for col in binary_cols:
        df[col].fillna(False, inplace=True)
    
    print(f"  {len(binary_cols)} columnas binarias rellenadas con False")
    
    return df

# -------------------------------------------------------------
# CREACIÓN DE SUBSETS TEMÁTICOS
# -------------------------------------------------------------

def create_consumption_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea subset para análisis de consumo diario.
    
    Returns:
    --------
    pd.DataFrame
        Subset con columnas relevantes
    """
    base_cols = ['submission_id', 'age', 'age_encoded','age_group', 
                 'gender', 'education',
                 'education_encoded', 'employment','employment_encoded', 
                 'children','children_encoded', 
                 'political_affiliation']
    consumption_cols = ['cups_per_day', 'cups_per_day_encoded', 'consumption_segment']
    
    cols = [col for col in base_cols + consumption_cols if col in df.columns]
    
    subset = df[cols].copy()

    # Rellenar NaN en children con '0' (sin hijos)
    if 'children' in subset.columns:
        subset['children'].fillna('0', inplace=True)
    if 'children_encoded' in subset.columns:
        subset['children_encoded'].fillna(0, inplace=True)

    print(f" Subset de consumo creado: {subset.shape}")
    
    return subset


def create_place_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea subset para análisis de lugares de consumo.
    """
    base_cols = ['submission_id', 'age', 'age_encoded', 'age_group', 
                 'gender', 'education', 
                 'education_encoded',                                 
                 'employment', 'employment_encoded',                   
                 'children', 'children_encoded',                       
                 'political_affiliation', 
                 'cups_per_day_encoded', 'consumption_segment']
    
    # Buscar columnas de lugares
    place_cols = [col for col in df.columns if 'Where do you typically drink coffee?' in col]
    
    cols = [col for col in base_cols + place_cols if col in df.columns]
    
    subset = df[cols].copy()
    
    # Rellenar NaN en children                 
    if 'children' in subset.columns:
        subset['children'].fillna('0', inplace=True)
    if 'children_encoded' in subset.columns:
        subset['children_encoded'].fillna(0, inplace=True)
    
    print(f" Subset de lugares creado: {subset.shape}")
    
    return subset


def create_home_brewing_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea subset para análisis de métodos de preparación en casa.
    """
    base_cols = ['submission_id', 'age', 'age_encoded', 'age_group', 
                 'gender', 'education', 
                 'education_encoded',                   
                 'employment', 'employment_encoded',             
                 'children', 'children_encoded',                   
                 'political_affiliation', 
                 'cups_per_day_encoded', 'consumption_segment']    
    
    # Buscar columnas de métodos de preparación
    brewing_cols = [col for col in df.columns if 'How do you brew coffee at home?' in col]
    
    cols = [col for col in base_cols + brewing_cols if col in df.columns]
    
    subset = df[cols].copy()
    
    # Rellenar NaN en children                              
    if 'children' in subset.columns:
        subset['children'].fillna('0', inplace=True)
    if 'children_encoded' in subset.columns:
        subset['children_encoded'].fillna(0, inplace=True)
    
    print(f" Subset de métodos en casa creado: {subset.shape}")
    
    return subset


def create_onthego_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea subset para análisis de negocios de compra on-the-go.
    """
    base_cols = ['submission_id', 'age', 'age_encoded', 'age_group',  
                 'gender', 'education', 
                 'education_encoded',                         
                 'employment', 'employment_encoded',             
                 'children', 'children_encoded',           
                 'political_affiliation', 
                 'cups_per_day_encoded', 'consumption_segment'] 
    
    # Buscar columnas de compra on-the-go
    purchase_cols = [col for col in df.columns if 'where do you typically purchase coffee?' in col.lower()]
    
    cols = [col for col in base_cols + purchase_cols if col in df.columns]
    
    subset = df[cols].copy()
    
    # Rellenar NaN en children             
    if 'children' in subset.columns:
        subset['children'].fillna('0', inplace=True)
    if 'children_encoded' in subset.columns:
        subset['children_encoded'].fillna(0, inplace=True)
    
    print(f" Subset de compras on-the-go creado: {subset.shape}")
    
    return subset


def create_dairy_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea subset para análisis de preferencias de lácteos.
    """
    base_cols = ['submission_id', 'age', 'age_encoded', 'age_group',  
                 'gender', 'education', 
                 'education_encoded',                            
                 'employment', 'employment_encoded',              
                 'children', 'children_encoded',                  
                 'political_affiliation', 
                 'cups_per_day_encoded', 'consumption_segment']  
    
    # Buscar columnas de lácteos
    dairy_cols = [col for col in df.columns if 'What kind of dairy do you add?' in col]
    
    cols = [col for col in base_cols + dairy_cols if col in df.columns]
    
    subset = df[cols].copy()
    
    # Rellenar NaN en children                               
    if 'children' in subset.columns:
        subset['children'].fillna('0', inplace=True)
    if 'children_encoded' in subset.columns:
        subset['children_encoded'].fillna(0, inplace=True)
    
    print(f" Subset de lácteos creado: {subset.shape}")
    
    return subset


def create_sweetener_subset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea subset para análisis de azucarantes.
    """
    base_cols = ['submission_id', 'age', 'age_encoded', 'age_group',  
                 'gender', 'education', 
                 'education_encoded',                         
                 'employment', 'employment_encoded',             
                 'children', 'children_encoded',                
                 'political_affiliation', 
                 'cups_per_day_encoded', 'consumption_segment']    
    
    # Buscar columnas de azucarantes
    sweetener_cols = [col for col in df.columns if 'What kind of sugar or sweetener do you add?' in col]
    
    cols = [col for col in base_cols + sweetener_cols if col in df.columns]
    
    subset = df[cols].copy()
    
    # Rellenar NaN en children                                  
    if 'children' in subset.columns:
        subset['children'].fillna('0', inplace=True)
    if 'children_encoded' in subset.columns:
        subset['children_encoded'].fillna(0, inplace=True)
    
    print(f" Subset de azucarantes creado: {subset.shape}")
    
    return subset

# ------------------------------------------------------------------------------
# PIPELINE COMPLETO DE LIMPIEZA
# -----------------------------------------------------------------------------

def full_cleaning_pipeline(filepath: str, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Pipeline completo de limpieza y transformación.
    
    Parameters:
    -----------
    filepath : str
        Ruta al CSV original
    output_path : str, optional
        Ruta para guardar el CSV limpio
        
    Returns:
    --------
    pd.DataFrame
        Dataset completamente limpio y transformado
    """
    print("="*80)
    print("INICIANDO PIPELINE DE LIMPIEZA")
    print("="*80)
    
    # 1. Cargar y limpieza inicial
    print("\n[1/7] Cargando datos...")
    df = load_and_initial_clean(filepath)
    
    # 2. Eliminar columnas con alta tasa de missing
    print("\n[2/7] Eliminando columnas con >95% missing...")
    df = remove_high_missing_columns(df, threshold=0.95)
    
    # 3. Estandarizar nombres
    print("\n[3/7] Estandarizando nombres de columnas...")
    df = standardize_column_names(df)
    
    # 4. Codificar variables ordinales
    print("\n[4/7] Codificando variables ordinales...")
    df = encode_all_ordinals(df)
    
    # 5. Crear variables derivadas
    print("\n[5/7] Creando variables derivadas...")
    if 'cups_per_day_encoded' in df.columns:
        df = create_consumption_segment(df)
    if 'age_encoded' in df.columns:
        df = create_age_groups(df)
    
    # 6. Imputar valores faltantes
    print("\n[6/7] Imputando valores faltantes...")
    df = impute_demographic_missing(df, strategy='unknown')
    df = fill_binary_columns_with_false(df)
    
    # 7. Guardar si se especifica path
    if output_path:
        print(f"\n[7/7] Guardando dataset limpio en {output_path}...")
        df.to_csv(output_path, index=False)
        print("✓ Guardado exitosamente")
    else:
        print("\n[7/7] No se especificó ruta de salida, omitiendo guardado")
    
    print("\n" + "="*80)
    print("LIMPIEZA COMPLETADA")
    print("="*80)
    print(f"Dataset final: {df.shape[0]} filas × {df.shape[1]} columnas")
    
    return df

def quick_summary(df: pd.DataFrame) -> None:
    """
    Imprime resumen rápido del dataset.
    """
    print("="*80)
    print("RESUMEN RÁPIDO")
    print("="*80)
    
    print(f"\nDimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Valores faltantes
    missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"\nValores faltantes: {missing:,} ({missing/total_cells*100:.2f}%)")
    
    # Tipos de datos
    print("\nTipos de datos:")
    print(df.dtypes.value_counts())
    
    # Demográficas
    if 'consumption_segment' in df.columns:
        print("\nSegmentos de consumo:")
        print(df['consumption_segment'].value_counts())
    
    if 'age_group' in df.columns:
        print("\nGrupos de edad:")
        print(df['age_group'].value_counts())


# ------------------------------------------------------------------------------
# EJEMPLO DE USO
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # Ejemplo de uso del pipeline completo
    
    INPUT_FILE = 'coffee-survey-project\coffee-survey-project\coffee-survey-full-dataset.csv'
    
    # Ejecutar limpieza completa
    df_clean = full_cleaning_pipeline(INPUT_FILE)
    
    # Ver resumen
    quick_summary(df_clean)
    
    # Crear subsets
    print("\n" + "="*80)
    print("CREANDO SUBSETS TEMÁTICOS")
    print("="*80 + "\n")
    
    subset_consumption = create_consumption_subset(df_clean)
    subset_places = create_place_subset(df_clean)
    subset_brewing = create_home_brewing_subset(df_clean)
    subset_onthego = create_onthego_subset(df_clean)
    subset_dairy = create_dairy_subset(df_clean)
    subset_sweetener = create_sweetener_subset(df_clean)
    
    print("\n Todos los subsets creados exitosamente")