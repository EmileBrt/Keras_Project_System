import hls4ml
from tensorflow import keras
import plotting
import os
os.environ['PATH'] = '/opt/Xilinx/Vivado/2019.2/bin:' + os.environ['PATH']

model = keras.models.load_model('fakemodel')
config = hls4ml.utils.config_from_keras_model(model, granularity='fakemodel')
print("-----------------------------------")
print("Configuration")
print(config)
print("-----------------------------------")

hls_model = hls4ml.converters.convert_from_keras_model(model,
                                                       hls_config=config,
                                                       output_dir='fakemodel_export/hls4ml_prj',
                                                       part="xc7z020clg484-1")

## hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)
hls_model.build(csim=False)
hls4ml.report.read_vivado_report('fakemodel_export/hls4ml_prj/')