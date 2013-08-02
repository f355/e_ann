%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module reads the architecture config and lets the supervisors spawn
%%% child processes accordingly. It then reads the input values and starts
%%% training the network. When it's done it will deliver the frozen weights
%%% that can be used for production training.
%%% @end
%%% Created :  19 June 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_training_handler).

-define(GLOBALERROR, 100.0).

-export([train_xor/0]).

%% Train a XOR with bias neurons.
train_xor() ->
    ICount = 2,
    HCount = 2,
    OCount = 1,
    LearningRate = 0.3,
    Momentum = 0.7,
    ErrorRate = 0.01,
    Inputs = read_training_data("priv/input_xor.txt"),
    Outputs = read_training_data("priv/output_xor.txt"),
    [{_,IBSup},{_, HBSup},{_,HSup},
     {_,OSup},{_,ISup}] = e_ann_network:get_sup_pids(),
    ISupFun = fun e_ann_input_neuron_sup:add_child/1,
    IWeightFun = fun e_ann_input_neuron:init_weights/2,
    HSupFun = fun e_ann_hidden_neuron_sup:add_child/1,
    HWeightFun = fun e_ann_hidden_neuron:init_weights/2,
    OSupFun = fun e_ann_output_neuron_sup:add_child/1,
    IBiasSupFun = fun e_ann_input_bias_neuron_sup:add_child/1,
    IBiasWeightFun = fun e_ann_input_bias_neuron:init_weights/2,
    HBiasSupFun = fun e_ann_hidden_bias_neuron_sup:add_child/1,
    HBiasWeightFun = fun e_ann_hidden_bias_neuron:init_weights/2,
    IL = e_ann_network:create_layer_with_random_weights(ICount, ISup, ISupFun,
                                                        IWeightFun, HCount),
    HL = e_ann_network:create_layer_with_random_weights(HCount, HSup , HSupFun,
                                                        HWeightFun, OCount),
    OL = e_ann_network:spawn_neurons(OCount, OSup, OSupFun, []),
    IBias = e_ann_network:spawn_bias_neuron(IBSup, HCount,
                                            IBiasSupFun, IBiasWeightFun),
    HBias = e_ann_network:spawn_bias_neuron(HBSup, OCount,
                                            HBiasSupFun, HBiasWeightFun),
    Layers = [IL, HL, OL, IBias, HBias],
    training_loop_xor(Inputs, Outputs, LearningRate, Momentum, ?GLOBALERROR,
                      ErrorRate,Layers),
    get_layer_weights(IL, HL, IBias, HBias).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Internal Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

training_loop_xor(_, _, _, _, GlobalError, ErrorRate, _)
  when GlobalError < ErrorRate ->
    training_complete;
training_loop_xor(Inputs, Outputs, LearningRate, Momentum,
              GlobalError, ErrorRate, Layers) when GlobalError > ErrorRate ->
    [Ilayer, Hlayer, Olayer, IBias, HBias] = Layers,
    IdealOutput = convert_to_integer(hd(Outputs)),
    set_ideal_output(Olayer, IdealOutput),
    Input = convert_to_integer(hd(Inputs)),
    feed_forward_input_layer_with_bias(Input, Ilayer, Hlayer, IBias),
    feed_forward_hidden_layer_with_bias(Hlayer, Olayer, HBias),
    backpropagation_output_layer_with_bias(Olayer, Hlayer, HBias),
    backpropagation_hidden_layer_with_bias(Hlayer, Ilayer, IBias),
    update_weights_input_layer_with_bias(Ilayer, IBias, LearningRate, Momentum),
    update_weights_hidden_layer_with_bias(Hlayer,HBias,LearningRate,Momentum),
    {ok, NewGlobalError} = e_ann_output_neuron:get_global_error(hd(Olayer)),
    %% infinite loop temporary
    training_loop_xor(Inputs, Outputs, LearningRate, Momentum,
                      NewGlobalError, ErrorRate, Layers).

feed_forward_input_layer_with_bias(Inputs, Ilayer, Layer, IBias) ->
    e_ann_network:set_inputs(Ilayer, Inputs),
    [ e_ann_input_neuron:feed_forward(Neuron, Layer) || Neuron <- Ilayer ],
    e_ann_input_bias_neuron:feed_forward(IBias, Layer).

feed_forward_hidden_layer_with_bias(Hlayer, Layer, HBias) ->
    [ e_ann_hidden_neuron:sum(Neuron) || Neuron <- Hlayer ],
    [ e_ann_hidden_neuron:activate_neuron(Neuron) || Neuron <- Hlayer ],
    [ e_ann_hidden_neuron:feed_forward(Neuron, Layer) || Neuron <- Hlayer ],
    e_ann_hidden_bias_neuron:feed_forward(HBias, Layer).

backpropagation_output_layer_with_bias(Olayer, Layer, Hbias) ->
    [ output_neuron_activation(Neuron) || Neuron <- Olayer ],
    [ calculate_output_neuron_delta(Neuron) || Neuron <- Olayer ],
    [ e_ann_output_neuron:backpropagate_with_bias(Neuron, Layer, Hbias)
     || Neuron <- Olayer ].

backpropagation_hidden_layer_with_bias(Hlayer, Ilayer, IBias) ->
    [ e_ann_hidden_neuron:backpropagate_with_bias(Neuron, Ilayer, IBias) ||
        Neuron <- Hlayer ].

update_weights_input_layer_with_bias(Ilayer, Ibias, LearningRate, Momentum) ->
    [ e_ann_input_neuron:update_weights(Neuron, LearningRate, Momentum) ||
        Neuron <- Ilayer ],
    e_ann_input_bias_neuron:update_weights(Ibias, LearningRate, Momentum).

update_weights_hidden_layer_with_bias(Hlayer, HBias, LearningRate, Momentum) ->
    [ e_ann_hidden_neuron:update_weights(Neuron, LearningRate, Momentum) ||
        Neuron <- Hlayer ],
    e_ann_hidden_bias_neuron:update_weights(HBias, LearningRate, Momentum).

output_neuron_activation(Neuron) ->
    e_ann_output_neuron:sum(Neuron),
    e_ann_output_neuron:activate_neuron(Neuron).

calculate_output_neuron_delta(Neuron) ->
    e_ann_output_neuron:calculate_error(Neuron),
    e_ann_output_neuron:calculate_node_delta(Neuron).

set_ideal_output([], []) ->
    ok;
set_ideal_output(Layer, Outputs) ->
    e_ann_output_neuron:set_ideal_output(hd(Layer), hd(Outputs)),
    set_ideal_output(tl(Layer), tl(Outputs)).

read_training_data(File) ->
    {ok, Bin} = file:read_file(File),
    SplitLines = binary:split(Bin, <<"\n">>, [global]),
    Lines = lists:delete([], [ binary_to_list(Line) || Line <- SplitLines ]),
    Inputs = [ re:split(L, ",", [{return, list}]) || L <- Lines ],
    lists:delete([], Inputs).

convert_to_integer(List) ->
    [ list_to_float(X) || X <-List ].

get_layer_weights(Ilayer, Hlayer, IBias, HBias) ->
    I_W = [ {input_neuron_weights, e_ann_input_neuron:get_weights(Neuron)} ||
              Neuron <- Ilayer ],
    H_W = [ {hidden_neuron_weights, e_ann_hidden_neuron:get_weights(Neuron)} ||
              Neuron <- Hlayer ],
    IBiasWeight = e_ann_input_bias_neuron:get_weights(IBias),
    HBiasWeight = e_ann_hidden_bias_neuron:get_weights(HBias),
    lists:flatten([I_W, H_W, {input_bias_weights, IBiasWeight},
     {hidden_bias_weight, HBiasWeight}]).
