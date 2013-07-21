%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module reads the architecture config and lets the supervisors spawn
%%% child processes accordingly. It then reads the input values and starts
%%% training the network.
%%% @end
%%% Created :  19 June 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_training_handler).

-define(MAINSUPERVISOR, e_ann_sup).
-compile([export_all]).

%% [[1.0,1.0],[0.0,1.0],[1.0,0.0],[0.0,0.0]] inputs
%% [[0.0],[1.0],[1.0],[0.0]]
train() ->
    Inputs = read_training_data("inputs.txt"),
    Outputs = read_training_data("outputs.txt"),
    [{_,IBSup},{_, HBSup}, {_,HSup},
     {_,OSup},{_,ISup}] = e_ann_training_handler:get_neuron_sup_pids(),
    ICount = 2,
    HCount = 2,
    OCount = 1,
    Momentum = 0.3,
    LearningRate = 0.7,
    Ilayer = e_ann_training_handler:create_input_layer(ICount, ISup, HCount),
    Hlayer = e_ann_training_handler:create_hidden_layer(HCount, HSup , OCount),
    Olayer = e_ann_training_handler:create_output_layer(OCount, OSup),
    IBias = input_bias(IBSup, 2),
    HBias = hidden_bias(HBSup, 1),
    Layers = [Ilayer, Hlayer, Olayer, IBias, HBias],
    ErrorRate = 0.01,
    GlobalError = 100.0,
    training_loop(Inputs, Outputs, LearningRate, Momentum,
                  GlobalError, ErrorRate, Layers).
    %% add_outputs_to_output_layer(Olayer, Outputs),
    %% feed_forward_input_layer_with_bias(Inputs, Ilayer, Hlayer, IBias),
    %% feed_forward_hidden_layer_with_bias(Hlayer, Olayer, HBias),
    %% backpropagation_output_layer_with_bias(Olayer, Hlayer, HBias),
    %% backpropagation_hidden_layer_with_bias(Hlayer, Ilayer, IBias),
    %% update_weights_input_layer_with_bias(Ilayer, IBias, LearningRate, Momentum),
    %% update_weights_hidden_layer_with_bias(Hlayer,HBias,LearningRate,Momentum).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Internal Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
training_loop(_, _, _, _, GlobalError, ErrorRate, _)
  when GlobalError < ErrorRate ->
    training_complete;
training_loop([], [], _, _, _, _, _) ->
    training_set_finished;
training_loop(Inputs, Outputs, LearningRate, Momentum,
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
    training_loop(Inputs, Outputs, LearningRate, Momentum,
                  NewGlobalError, ErrorRate, Layers).

feed_forward_input_layer_with_bias(Inputs, Ilayer, Layer, IBias) ->
    add_inputs_to_input_layer(Ilayer, Inputs),
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

create_output_layer(OCount, OSup) ->
    get_output_neurons(OCount, OSup, []).

create_hidden_layer(HCnt, HSup , OCnt) ->
    HiddenNeuronPids = get_hidden_neurons(HCnt, HSup, []),
    [ e_ann_hidden_neuron:init_weights(Pid, OCnt) || Pid <- HiddenNeuronPids ],
    HiddenNeuronPids.

create_input_layer(ICount, ISup, HCount) ->
    InputNeuronPids = get_input_neurons(ICount, ISup, []),
    [ e_ann_input_neuron:init_weights(Pid, HCount) || Pid <- InputNeuronPids ],
    InputNeuronPids.

get_neuron_sup_pids() ->
    [{_, IBSup, _, _}, {_, HBSup,_ ,_}, {_, HSup, _, _},
     {_, OSup, _, _}, {_, ISup, _, _}] =
        supervisor:which_children(?MAINSUPERVISOR),
    [{input_bias_sup, IBSup},{hidden_bias_sup, HBSup}, {hidden_sup, HSup},
     {output_sup, OSup},{input_sup, ISup}].

input_bias(Sup, Count) ->
    {ok, Pid} = e_ann_input_bias_neuron_sup:add_child(Sup),
    e_ann_input_bias_neuron:init_weights(Pid, Count),
    Pid.

hidden_bias(Sup, Count) ->
    {ok, Pid} = e_ann_hidden_bias_neuron_sup:add_child(Sup),
    e_ann_hidden_bias_neuron:init_weights(Pid, Count),
    Pid.

get_input_neurons(0, _, Acc) ->
    Acc;
get_input_neurons(ICount, ISup, Acc) ->
    {ok, Pid} = e_ann_input_neuron_sup:add_child(ISup),
    NewCount = ICount - 1,
    Acc2 = [Pid | Acc],
    get_input_neurons(NewCount, ISup, Acc2).

get_hidden_neurons(0, _, Acc) ->
    Acc;
get_hidden_neurons(HCount, HSup, Acc) ->
    {ok, Pid} = e_ann_hidden_neuron_sup:add_child(HSup),
    NewCount = HCount - 1,
    Acc2 = [Pid | Acc],
    get_hidden_neurons(NewCount, HSup, Acc2).

get_output_neurons(0, _, Acc) ->
    Acc;
get_output_neurons(OCount, OSup, Acc) ->
    {ok, Pid} = e_ann_output_neuron_sup:add_child(OSup),
    NewCount = OCount - 1,
    Acc2 = [Pid | Acc],
    get_output_neurons(NewCount, OSup, Acc2).

add_inputs_to_input_layer([], []) ->
    ok;
add_inputs_to_input_layer(Layer, Inputs) ->
    e_ann_input_neuron:add_input(hd(Layer), hd(Inputs)),
    add_inputs_to_input_layer(tl(Layer), tl(Inputs)).

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
