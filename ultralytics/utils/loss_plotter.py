import pandas as pd
import numpy as np

only3D = "/storage/group/deepscenario/jonathan_for_johannes/results_csvs/yolov10-3Db_only3D_2"
no3D = "/storage/group/deepscenario/jonathan_for_johannes/results_csvs/yolov10-3Db_wo3Dtal_3"
both = "/storage/group/deepscenario/jonathan_for_johannes/results_csvs/yolov10-3Db_weightedDist_21"

only3D = pd.read_csv(only3D)[""]
