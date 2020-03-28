# configurations for raw data (running logs) processing.
# CHECK this configuration before running raw pre-processing program.
# author rovo98

# specifying the location of the running logs.
DEFAULT_RAW_DATA_PATH = '../../generated-logs'

# !!the maximum length of the observation of the logs to be loaded!!
OBSERVATION_LENGTH = 100

# location to save the processed running logs.
GENERATED_LOGS_LOC = '../dataset'

# by default, was the filename to be loaded.
# NOT NEED TO CHANGE (Recommended)
DEFAULT_GENERATED_LOG_FILE_NAME = ''

# location to save configuration for encoding new come running logs.
GENERATED_ENCODING_CONFIG_LOC = '../encoding-configs'
