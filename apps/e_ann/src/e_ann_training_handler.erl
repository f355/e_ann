%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module trains the network to find optimal weights.
%%% When it's done it will deliver the frozen weights that can be
%%% used for production training.
%%% @end
%%% Created :  19 June 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_training_handler).

-define(GLOBALERROR, 100.0).
-define(ERRORRATE, 0.01).
-define(LEARNINGRATE, 0.01).
-define(MOMENTUM, 0.7).

-define(GAWINPUTS, "Netflix_inputs1.csv").
-define(GAWOUTPUTS, "Netflix_output1.csv").

-compile([export_all]).
-export([train_xor/0]).
-export([train_gaw_data/1]).

train_gaw_data(SecondsToRun) ->
    ICount = 11,
    HCount = 6,
    OCount = 1,
    Inputs = tl(read_training_data(?GAWINPUTS)),
    Outputs = tl(read_training_data(?GAWOUTPUTS)),
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
    erlang:send_after(SecondsToRun*1000, self(), {stop_now}),
    training_loop_gaw(Inputs, Outputs, [], [], ?LEARNINGRATE,
                      ?MOMENTUM, ?GLOBALERROR, ?ERRORRATE, Layers),
    get_layer_weights(IL, HL, IBias, HBias).

%% Train a XOR with bias neurons.
train_xor() ->
    ICount = 2,
    HCount = 2,
    OCount = 1,
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
    training_loop_xor(Inputs, Outputs, ?LEARNINGRATE, ?MOMENTUM, ?GLOBALERROR,
                      ?ERRORRATE, Layers),
    get_layer_weights(IL, HL, IBias, HBias).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Internal Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% training_loop_gaw(_, _, _, _, _GlobalError, _ErrorRate, _) ->
%%     training_complete;
%% training_loop_gaw(_, _, _, _, GlobalError, ErrorRate, _, _)
%%   when GlobalError < ErrorRate ->
%%     training_complete;
training_loop_gaw([], [], IAcc, OAcc, LearningRate, Momentum,
                  GlobalError, ErrorRate, Layers) ->
    training_loop_gaw(lists:reverse(IAcc), lists:reverse(OAcc), [], [],
                      LearningRate, Momentum, GlobalError, ErrorRate, Layers);
training_loop_gaw([I | Inputs], [O | Outputs], IAcc, OAcc, LearningRate, Momentum,
                  _GlobalError, ErrorRate, Layers) ->
    [Ilayer, Hlayer, Olayer, IBias, HBias] = Layers,
    IdealOutput = convert_to_integer(O),
    set_ideal_output(Olayer, IdealOutput),
    Input = convert_to_integer(I),
    feed_forward_input_layer_with_bias(Input, Ilayer, Hlayer, IBias),
    feed_forward_hidden_layer_with_bias(Hlayer, Olayer, HBias),
    backpropagation_output_layer_with_bias(Olayer, Hlayer, HBias),
    backpropagation_hidden_layer_with_bias(Hlayer, Ilayer, IBias),
    update_weights_input_layer_with_bias(Ilayer, IBias, LearningRate, Momentum),
    update_weights_hidden_layer_with_bias(Hlayer,HBias,LearningRate,Momentum),
    {ok, NewGlobalError} = e_ann_output_neuron:get_global_error(hd(Olayer)),
    receive
        {stop_now} ->
            training_complete
    after 0 ->
            training_loop_gaw(Inputs, Outputs, [I | IAcc], [O | OAcc],
                              LearningRate, Momentum, NewGlobalError,
                              ErrorRate, Layers)
    end.


training_loop_xor(_, _, _, _, GlobalError, ErrorRate, _)
  when GlobalError < ErrorRate ->
    training_complete;
training_loop_xor(Inputs, Outputs, LearningRate, Momentum,
              GlobalError, ErrorRate, Layers) when GlobalError > ErrorRate ->
    [Ilayer, Hlayer, Olayer, IBias, HBias] = Layers,
    IdealOutput = convert_to_integer(hd(Outputs)),
    set_ideal_output(Olayer, IdealOutput),
    Input = convert_to_integer(hd(Inputs)),
    reset_inputs(Hlayer, Olayer),
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

reset_inputs(Hlayer, Olayer) ->
    [e_ann_hidden_neuron:reset_input(Neuron) || Neuron <- Hlayer],
    [e_ann_output_neuron:reset_input(Neuron) || Neuron <- Olayer],
    ok.

set_ideal_output([], []) ->
    ok;
set_ideal_output(Layer, Outputs) ->
    e_ann_output_neuron:set_ideal_output(hd(Layer), hd(Outputs)),
    set_ideal_output(tl(Layer), tl(Outputs)).

read_training_data(File) ->
    {ok, Bin} = file:read_file(File),
    SplitLines = binary:split(Bin, <<"\n">>, [global]),
    Lines = lists:delete([], [ binary_to_list(Line) || Line <- SplitLines ]),
    Inputs = [ re:split(L, "\t", [{return, list}]) || L <- Lines ],
    lists:delete([], Inputs).

read_gaw_training_data(File) ->
    {ok, Bin} = file:read_file(File),
    binary:split(Bin, <<"\n">>, [global]).

read_gaw_training_data_line(Line) ->
    Values = binary:split(Line, <<";">>, [global]),
    lists:map(fun(X) -> convert_percentages_to_float(X) end, Values).

convert_percentages_to_float(Percentage) ->
    Regex = re:replace(Percentage, "[%]", "", [{return,list}]),
    Num = convert_list(Regex),
    Number = convert_number(Num / 100),
    list_to_binary(Number).

convert_to_integer(List) ->
    [ maybe_list_to_float(X) || X <-List ].

convert_number(Number) when is_integer(Number) ->
    integer_to_list(Number);
convert_number(Number) when is_float(Number) ->
    float_to_list(Number).

convert_list(List) ->
    case string:to_float(List) of
        {error,_} ->
            list_to_integer(List);
        {Float, _ } ->
            Float
    end.


get_layer_weights(Ilayer, Hlayer, IBias, HBias) ->
    I_W = [ {input_neuron_weights, e_ann_input_neuron:get_weights(Neuron)} ||
              Neuron <- Ilayer ],
    H_W = [ {hidden_neuron_weights, e_ann_hidden_neuron:get_weights(Neuron)} ||
              Neuron <- Hlayer ],
    IBiasWeight = e_ann_input_bias_neuron:get_weights(IBias),
    HBiasWeight = e_ann_hidden_bias_neuron:get_weights(HBias),
    lists:flatten([I_W, H_W, {input_bias_weights, IBiasWeight},
     {hidden_bias_weight, HBiasWeight}]).

maybe_list_to_float(Str) ->
    case catch list_to_float(Str) of
        F when is_float(F) ->
            F;
        _ ->
            list_to_integer(Str)/1.0
    end.
