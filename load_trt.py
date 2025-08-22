#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import argparse
import timeit
import pandas as pd
import tensorrt as trt


def load_trt_model(engine_path) -> tuple[trt.ICudaEngine, trt.IExecutionContext, float]:
    logger = trt.Logger(trt.Logger.ERROR)
    engine, context = None, None
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        start_time = timeit.default_timer()
        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        load_time = timeit.default_timer() - start_time
    return engine, context, load_time

def main(args):
    # model_family, model_variant, weights, input_shape = model_config[args.model_id]
    engine_path = f"trt_models/{args.model_id}.trt"
    if not os.path.exists(engine_path):
        print(f"TRT Engine file {engine_path} does not exist")
        return
    engine, context, load_time = load_trt_model(engine_path)
    print(engine)
    print(context)
    print(f"Model {args.model_id} loaded in {load_time * 1000:.2f} ms")

    # df = pd.read_csv(args.output_file, index_col=0)
    # if 'trt_load_time_ms' not in df.columns:
    #     df['trt_load_time_ms'] = 0.0
    
    # df.loc[args.model_id, 'trt_load_time_ms'] = load_time * 1000
    # df.to_csv(args.output_file, index=True)
    # print(f"Updated row in output csv: {df.loc[args.model_id]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Profiling on GPU')
    parser.add_argument("-m", "--model_id", type=str, default=None, help="Model ID to run")
    parser.add_argument("-o", "--output_file", type=str, default="profiles.csv", help="Output CSV file")
    args = parser.parse_args()
    main(args)
    
    