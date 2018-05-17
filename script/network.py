def net_proto(self):
    conv_weight_filler = layers.Filler("gaussian", 0.01)
    bias_filler0 = layers.Filler("constant", 0.0)
    bias_filler1 = layers.Filler("constant", 0.1)
    bias_filler5 = layers.Filler("constant", 0.5)

    # same deploy structure as in deploy_demo.prototxt
    net_layers = [
    # saliency path
        layers.Convolution("conv1", bottoms=["data"], param_lr_mults=[0, 0],
            param_decay_mults=[1, 0], kernel_dim=(11, 11),
            stride=4, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=96),
        layers.ReLU(name="relu1", bottoms=["conv1"], tops=["conv1"]),
        layers.Pooling(name="pool1", bottoms=["conv1"], kernel_size=3, stride=2),
        layers.LRN(name="norm1", bottoms=["pool1"], tops=["norm1"], local_size=5, alpha=0.0001, beta=0.75),

        layers.Convolution(name="conv2", bottoms=["norm1"], param_lr_mults=[0, 0],
            param_decay_mults=[1, 0], kernel_dim=(5, 5),
            pad=2, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256),
        layers.ReLU(name="relu2", bottoms=["conv2"], tops=["conv2"]),
        layers.Pooling(name="pool2", bottoms=["conv2"], kernel_size=3, stride=2),
        layers.LRN(name="norm2", bottoms=["pool2"], tops=["norm2"], local_size=5, alpha=0.0001, beta=0.75),

        layers.Convolution(name="conv3", bottoms=["norm2"], param_lr_mults=[0.1, 0.2],
            param_decay_mults=[1, 0], kernel_dim=(3, 3),
            pad=1, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=384),
        layers.ReLU(name="relu3", bottoms=["conv3"], tops=["conv3"]),

        layers.Convolution(name="conv4", bottoms=["conv3"], param_lr_mults=[0.1, 0.2],
            param_decay_mults=[1, 0], kernel_dim=(3, 3),
            pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=384),
        layers.ReLU(name="relu4", bottoms=["conv4"], tops=["conv4"]),

        layers.Convolution(name="conv5", bottoms=["conv4"], param_lr_mults=[0.1, 0.2],
            param_decay_mults=[1, 0], kernel_dim=(3, 3),
            pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256),
        layers.ReLU(name="relu5", bottoms=["conv5"], tops=["conv5"]),

        layers.Convolution(name="conv5_red", bottoms=["conv5"], param_lr_mults=[1.0, 2.0],
            param_decay_mults=[1, 0], kernel_dim=(1, 1),
            weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=1),
        layers.ReLU(name="relu5_red", bottoms=["conv5_red"], tops=["conv5_red"]),

        # gaze path
        layers.Convolution("conv1_face", bottoms=["face"], param_lr_mults=[0, 0],
            param_decay_mults=[1, 0], kernel_dim=(11, 11),
            stride=4, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=96),
        layers.ReLU(name="relu1_face", bottoms=["conv1_face"], tops=["conv1_face"]),
        layers.Pooling(name="pool1_face", bottoms=["conv1_face"], kernel_size=3, stride=2),
        layers.LRN(name="norm1_face", bottoms=["pool1_face"], tops=["norm1_face"], local_size=5, alpha=0.0001, beta=0.75),

        layers.Convolution(name="conv2_face", bottoms=["norm1_face"], param_lr_mults=[0, 0],
            param_decay_mults=[1, 0], kernel_dim=(5, 5),
            pad=2, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256),
        layers.ReLU(name="relu2_face", bottoms=["conv2_face"], tops=["conv2_face"]),
        layers.Pooling(name="pool2_face", bottoms=["conv2_face"], kernel_size=3, stride=2),
        layers.LRN(name="norm2_face", bottoms=["pool2_face"], tops=["norm2_face"], local_size=5, alpha=0.0001, beta=0.75),

        layers.Convolution(name="conv3_face", bottoms=["norm2_face"], param_lr_mults=[0.2, 0.4],
            param_decay_mults=[1, 0], kernel_dim=(3, 3),
            pad=1, weight_filler=conv_weight_filler, bias_filler=bias_filler0, num_output=384),
        layers.ReLU(name="relu3_face", bottoms=["conv3_face"], tops=["conv3_face"]),

        layers.Convolution(name="conv4_face", bottoms=["conv3_face"], param_lr_mults=[0.2, 0.4],
            param_decay_mults=[1, 0], kernel_dim=(3, 3),
            pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=384),
        layers.ReLU(name="relu4_face", bottoms=["conv4_face"], tops=["conv4_face"]),

        layers.Convolution(name="conv5_face", bottoms=["conv4_face"], param_lr_mults=[0.2, 0.4],
            param_decay_mults=[1, 0], kernel_dim=(3, 3),
            pad=1, group=2, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=256),
        layers.ReLU(name="relu5_face", bottoms=["conv5_face"], tops=["conv5_face"]),
        layers.Pooling(name="pool5_face", bottoms=["conv5_face"], kernel_size=3, stride=2),

        layers.InnerProduct(name="fc6_face", bottoms=["pool5_face"], tops=["fc6_face"], param_lr_mults=[1, 2],
            param_decay_mults=[1, 0],
            weight_filler=layers.Filler("gaussian", 0.5),
            bias_filler=bias_filler5, num_output=500),
        layers.ReLU(name="relu6_face", bottoms=["fc6_face"], tops=["fc6_face"]),

        layers.Flatten(name="eyes_grid_flat", bottoms=["eyes_grid"], tops=["eyes_grid_flat"]),
        layers.Power(name="eyes_grid_mult", bottoms=["eyes_grid_flat"], tops=["eyes_grid_mult"],
                     power=1, scale=24, shift=0),
        layers.Concat(name="face_input", bottoms=["fc6_face", "eyes_grid_mult"], tops=["face_input"], axis=1),

        layers.InnerProduct(name="fc7_face", bottoms=["face_input"], tops=["fc7_face"], param_lr_mults=[1, 2],
                            param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                            bias_filler=bias_filler5, num_output=400),
        layers.ReLU(name="relu7_face", bottoms=["fc7_face"], tops=["fc7_face"]),

        layers.InnerProduct(name="fc8_face", bottoms=["fc7_face"], tops=["fc8_face"], param_lr_mults=[1, 2],
                            param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                            bias_filler=bias_filler5, num_output=200),
        layers.ReLU(name="relu8_face", bottoms=["fc8_face"], tops=["fc8_face"]),

        layers.InnerProduct(name="importance_no_sigmoid", bottoms=["fc8_face"], tops=["importance_no_sigmoid"], param_lr_mults=[0.2, 0.0],
                            param_decay_mults=[1.0, 0.0],
                            weight_filler=layers.Filler("gaussian", 0.01), num_output=169),

        layers.Sigmoid(name="importance_map_prefilter", bottoms=["importance_no_sigmoid"], tops=["importance_map_prefilter"]),
        layers.Reshape('importance_map_reshape', (1, 1, 13, 13), bottoms=['importance_map_prefilter'], tops=["importance_map_reshape"]),

        layers.Convolution(name="importance_map", bottoms=["importance_map_reshape"], param_lr_mults=[0.0, 0.0],
            param_decay_mults=[1.0, 0.0], kernel_dim=(3, 3),
            pad=1, stride=1, weight_filler=conv_weight_filler, bias_filler=bias_filler1, num_output=1),
        layers.Eltwise(name="fc_7", bottoms=["conv5_red", "importance_map"], tops=["fc_7"], operation="PROD"),

        # shifted grids
        layers.InnerProduct(name="fc_0_0", bottoms=["fc_7"], tops=["fc_0_0"], param_lr_mults=[1, 2],
                            param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                            bias_filler=bias_filler0, num_output=25),
        layers.InnerProduct(name="fc_1_0", bottoms=["fc_7"], tops=["fc_1_0"], param_lr_mults=[1, 2],
                            param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                            bias_filler=bias_filler0, num_output=25),
        layers.InnerProduct(name="fc_m1_0", bottoms=["fc_7"], tops=["fc_m1_0"], param_lr_mults=[1, 2],
                            param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                            bias_filler=bias_filler0, num_output=25),
        layers.InnerProduct(name="fc_0_1", bottoms=["fc_7"], tops=["fc_0_1"], param_lr_mults=[1, 2],
                            param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                            bias_filler=bias_filler0, num_output=25),
        layers.InnerProduct(name="fc_0_m1", bottoms=["fc_7"], tops=["fc_0_m1"], param_lr_mults=[1, 2],
                            param_decay_mults=[1, 0], weight_filler=layers.Filler("gaussian", 0.01),
                            bias_filler=bias_filler0, num_output=25),
        layers.Reshape('fc_0_0_reshape', (5, 5), bottoms=['fc_0_0'], tops=["fc_0_0_reshape"]),
        layers.Reshape('fc_1_0_reshape', (5, 5), bottoms=['fc_1_0'], tops=["fc_1_0_reshape"]),
        layers.Reshape('fc_m1_0_reshape', (5, 5), bottoms=['fc_m1_0'], tops=["fc_m1_0_reshape"]),
        layers.Reshape('fc_0_1_reshape', (5, 5), bottoms=['fc_0_1'], tops=["fc_0_1_reshape"]),
        layers.Reshape('fc_0_m1_reshape', (5, 5), bottoms=['fc_0_m1'], tops=["fc_0_m1_reshape"])
        ]
    return net_layers
