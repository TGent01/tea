from keras.models import Model
from keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, TimeDistributed,
)
from keras.regularizers import l2 as l2_reg_fn

from config import CONFIG

SEQUENCE_LENGTH = CONFIG['sequence_length']
IMAGE_HEIGHT = IMAGE_WIDTH = CONFIG['image_size']
CANALES = 3


def _bloque_conv(x, filtros, pool_size, nombre, con_dropout, dropout_rate, usar_bn, l2_reg):
    regularizador = l2_reg_fn(l2_reg) if l2_reg > 0 else None

    if usar_bn:
        # Patrón Conv-BN-ReLU (sin bias, redundante cuando hay BatchNorm justo después)
        x = TimeDistributed(
            Conv2D(filtros, (3, 3), padding="same", use_bias=False, kernel_regularizer=regularizador),
            name=f"{nombre}_conv",
        )(x)
        x = TimeDistributed(BatchNormalization(), name=f"{nombre}_bn")(x)
        x = TimeDistributed(Activation("relu"), name=f"{nombre}_relu")(x)
    else:
        x = TimeDistributed(
            Conv2D(filtros, (3, 3), padding="same", activation="relu", kernel_regularizer=regularizador),
            name=f"{nombre}_conv",
        )(x)

    x = TimeDistributed(MaxPooling2D(pool_size), name=f"{nombre}_pool")(x)
    if con_dropout:
        x = TimeDistributed(Dropout(dropout_rate), name=f"{nombre}_dropout")(x)
    return x


def crear_cnn_extractor(usar_bn=False, dropout=0.3, l2_reg=0.0):
    """Sub-modelo espacial (capa CNN), completamente independiente de la LSTM.

    Recibe una secuencia de frames de forma
    (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CANALES) y devuelve, para
    cada frame, un vector de características de tamaño 64
    (forma de salida: (SEQUENCE_LENGTH, 64)).

    Parámetros (para poder comparar configuraciones de forma controlada,
    ver entrenar_multiples_semillas.py):
        usar_bn: si True, agrega BatchNormalization después de cada
            convolución (patrón Conv-BN-ReLU). Por defecto False, ya que en
            pruebas empíricas con este dataset dio resultados más inestables.
        dropout: probabilidad de Dropout en los bloques 1-3 (el bloque 4 no
            lleva Dropout, al ser el último antes de aplanar).
        l2_reg: fuerza de la regularización L2 sobre los pesos de las
            convoluciones. 0.0 la desactiva (valor por defecto).
    """
    entrada = Input(
        shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, CANALES),
        name="frames_entrada",
    )

    x = _bloque_conv(entrada, 16, (4, 4), "bloque1", True, dropout, usar_bn, l2_reg)
    x = _bloque_conv(x, 32, (4, 4), "bloque2", True, dropout, usar_bn, l2_reg)
    x = _bloque_conv(x, 64, (2, 2), "bloque3", True, dropout, usar_bn, l2_reg)
    x = _bloque_conv(x, 64, (2, 2), "bloque4", False, dropout, usar_bn, l2_reg)

    # Aplanar cada frame (1x1x64) a un vector de 64 características
    salida = TimeDistributed(Flatten(), name="vector_caracteristicas")(x)

    return Model(inputs=entrada, outputs=salida, name="CNN_extractor_espacial")


if __name__ == "__main__":
    extractor = crear_cnn_extractor()
    extractor.summary()
