from flask_wtf import FlaskForm
from wtforms import BooleanField, FileField, FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange
from wtforms import ValidationError

class IndividualPredictionForm(FlaskForm):
    # Sección 1: Información Básica (C1–C3)
    age               = FloatField('Edad (años)', validators=[DataRequired(), NumberRange(15, 50)])
    bmi               = FloatField('Índice de Masa Corporal (BMI)', validators=[DataRequired(), NumberRange(15, 50)])
    gestational_age   = FloatField('Edad gestacional al parto (semanas)', validators=[DataRequired(), NumberRange(20, 42)])
    
    # Sección 2: Historial Obstétrico (C4–C6)
    gravidity         = IntegerField('Número de embarazos (Gravidez)', validators=[DataRequired(), NumberRange(0, 20)])
    parity            = IntegerField('Número de partos (Paridad)', validators=[DataRequired(), NumberRange(0, 15)])
    initial_symptoms  = SelectField('Síntomas iniciales', choices=[
                            ('0','Edema'),
                            ('1','Hipertensión'),
                            ('2','FGR (Restricción del crecimiento fetal)')
                          ], validators=[DataRequired()])
    
    # Sección 3: Tiempos de aparición (C7–C14)
    ios_onset_age         = IntegerField('Edad de inicio de IOS (semanas)', validators=[DataRequired(), NumberRange(0, 50)])
    ios_to_delivery       = IntegerField('Días de IOS a parto', validators=[DataRequired(), NumberRange(0, 300)])
    htn_onset_age         = IntegerField('Edad de inicio de hipertensión (semanas)', validators=[DataRequired(), NumberRange(0, 50)])
    htn_to_delivery       = IntegerField('Días de hipertensión a parto', validators=[DataRequired(), NumberRange(0, 300)])
    edema_onset_age       = IntegerField('Edad de inicio de edema (semanas)', validators=[DataRequired(), NumberRange(0, 50)])
    edema_to_delivery     = IntegerField('Días de edema a parto', validators=[DataRequired(), NumberRange(0, 300)])
    proteinuria_onset_age = IntegerField('Edad de inicio de proteinuria (semanas)', validators=[DataRequired(), NumberRange(0, 50)])
    proteinuria_to_delivery = IntegerField('Días de proteinuria a parto', validators=[DataRequired(), NumberRange(0, 300)])
    
    # Sección 4: Tratamiento y antecedentes (C15–C17)
    expectant_treatment   = BooleanField('Expectant treatment')   # C15
    antihypert_before     = BooleanField('Antihipertensivo previo')  # C16
    past_history          = SelectField('Antecedentes', choices=[
                            ('0','No'),
                            ('1','Hipertensión'),
                            ('2','PCOS')
                          ], validators=[DataRequired()])  # C17
    
    # Sección 5: Signos Vitales (C18–C19)
    max_systolic       = IntegerField('Presión sistólica máxima (mmHg)', validators=[DataRequired(), NumberRange(70, 250)])
    max_diastolic      = IntegerField('Presión diastólica máxima (mmHg)', validators=[DataRequired(), NumberRange(40, 150)])
    
    # Sección 6: Pruebas de Laboratorio básicas (C22–C25)
    bnp                = FloatField('Valor máximo de BNP (pg/mL)', validators=[DataRequired(), NumberRange(0, 5000)])
    creatinine         = FloatField('Creatinina máxima (mg/dL)', validators=[DataRequired(), NumberRange(0.1, 10)])
    uric_acid          = FloatField('Ácido úrico máximo (mg/dL)', validators=[DataRequired(), NumberRange(0, 20)])     # C24
    proteinuria        = FloatField('Proteinuria máxima (mg/24h)', validators=[DataRequired(), NumberRange(0, 10000)])
    total_protein      = FloatField('Proteína total máxima (mg/dL)', validators=[DataRequired(), NumberRange(0, 20)])  # C26
    
    # Sección 7: Laboratorio extendido (C27–C30)
    albumin            = FloatField('Albúmina máxima (g/dL)', validators=[DataRequired(), NumberRange(0, 10)])       # C27
    alt                = FloatField('ALT máxima (U/L)', validators=[DataRequired(), NumberRange(0, 300)])            # C28
    ast                = FloatField('AST máxima (U/L)', validators=[DataRequired(), NumberRange(0, 300)])            # C29
    platelet           = FloatField('Plaquetas máximas (10³/μL)', validators=[DataRequired(), NumberRange(10, 500)])
    
    # Sección 8: Datos de parto (C20–C21)
    reasons_delivery   = SelectField('Razones para parto', choices=[
                            ('0','HELLP Syndrome'),
                            ('1','Fetal distress'),
                            ('2','Organ dysfunction'),
                            ('3','Uncontrolled hypertension'),
                            ('4','Edema'),
                            ('5','FGR')
                          ], validators=[DataRequired()])
    mode_of_delivery   = SelectField('Modo de parto', choices=[
                            ('0','CS'),
                            ('1','Odinopoeia')
                          ], validators=[DataRequired()])
    
    # Sección 9: Fetal weight (C31 — objetivo; opcional si se quiere simular)
    fetal_weight       = SelectField('Peso fetal', choices=[
                            ('0','Normal'),
                            ('1','FGR')
                          ], validators=[DataRequired()])
    
    # Modelo a utilizar
    model_choice = SelectField('Modelo a utilizar', 
                             choices=[
                                 ('logistic', 'Regresión Logística'),
                                 ('neural', 'Red Neuronal'),
                                 ('svm', 'Máquina de Soporte Vectorial'),
                                 ('fuzzy', 'Mapa Difuso')
                             ],
                             validators=[DataRequired()])
    
    submit = SubmitField('Predecir')

class BatchPredictionForm(FlaskForm):
    file = FileField('Archivo de datos', validators=[DataRequired()])
    model_choice = SelectField('Modelo a utilizar',
                             choices=[
                                 ('logistic', 'Regresión Logística'),
                                 ('neural', 'Red Neuronal'),
                                 ('svm', 'Máquina de Soporte Vectorial'),
                                 ('fuzzy', 'Mapa Difuso')
                             ],
                             validators=[DataRequired()])
    submit = SubmitField('Procesar')