from keras.models import Model

from cnn_extractor import crear_cnn_extractor
from modelo_lstm import crear_lstm_clasificador


def crear_modelo(num_clases, usar_bn=False, dropout_cnn=0.3, dropout_lstm=0.3, l2_reg=0.0):
    """Combina el extractor CNN (espacial, cnn_extractor.py) y el
    clasificador LSTM (temporal, modelo_lstm.py) en un único modelo
    entrenable de punta a punta.

    Ambos submodelos son independientes entre sí; este archivo es el único
    punto donde se conectan.

    Los parámetros por defecto corresponden a la configuración "liviana"
    (sin BatchNormalization) que dio mejores resultados empíricos con este
    dataset. Se pueden ajustar para comparar configuraciones de forma
    controlada -- ver entrenar_multiples_semillas.py.
    """
    extractor = crear_cnn_extractor(usar_bn=usar_bn, dropout=dropout_cnn, l2_reg=l2_reg)
    clasificador = crear_lstm_clasificador(num_clases, dropout=dropout_lstm, l2_reg=l2_reg)

    salida = clasificador(extractor.output)
    return Model(inputs=extractor.input, outputs=salida, name="CNN_LSTM_TEA")


if __name__ == "__main__":
    modelo = crear_modelo(num_clases=3)
    modelo.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    modelo.summary()
