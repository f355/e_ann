%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module reads the architecture config and lets the supervisors spawn
%%% child processes accordingly. It then reads the input values and starts
%%% training the network.
%%% @end
%%% Created :  19 May 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_training_handler).

-define(MAINSUPERVISOR, e_ann_sup).
-compile([export_all]).

train() ->
    [{_,IBSup},{_, HBSup}, {_,HSup},
     {_,OSup},{_,ISup}] = e_ann_training_handler:get_neuron_sup_pids(),
    ICount = 2,
    HCount = 2,
    OCount = 1,
    Ilayer = e_ann_training_handler:create_input_layer([1.0,0.0],
                                                       ICount , ISup, HCount),
    Hlayer = e_ann_training_handler:create_hidden_layer(HCount ,HSup , OCount),
    Olayer = e_ann_training_handler:create_output_layer([1.0], OCount, OSup),
    IBias = input_bias(IBSup, 2),
    HBias = hidden_bias(HBSup, 1),
    feed_forward_input_layer_with_bias(Ilayer, Hlayer, IBias),
    feed_forward_hidden_layer_with_bias(Hlayer, Olayer, HBias),
    Olayer.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Internal Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Create funs for feed_forward and back propagation.
feed_forward_input_layer_with_bias(Ilayer, Layer, IBias) ->
    [ e_ann_input_neuron:feed_forward(Neuron, Layer) || Neuron <- Ilayer ],
    e_ann_input_bias_neuron:feed_forward(IBias, Layer).

feed_forward_hidden_layer_with_bias(Hlayer, Layer, HBias) ->
    [ e_ann_hidden_neuron:sum(Neuron) || Neuron <- Hlayer ],
    [ e_ann_hidden_neuron:activate_neuron(Neuron) || Neuron <- Hlayer ],
    [ e_ann_hidden_neuron:feed_forward(Neuron, Layer) || Neuron <- Hlayer ],
    e_ann_hidden_bias_neuron:feed_forward(HBias, Layer).

backpropagation_output_layer(_Olayer, _Layer) ->
  ok.

output_neuron_activation(Neuron) ->
    e_ann_output_neuron:sum(Neuron),
    e_ann_output_neuron:activate_neuron(Neuron).

calculate_output_neuron_delta(Neuron) ->
    e_ann_output_neuron:calculate_error(Neuron),
    e_ann_output_neuron:calculate_node_delta(Neuron).

create_output_layer(Ideal, OCount, OSup) ->
    get_output_neurons(OCount, OSup, Ideal, []).

create_hidden_layer(HCount, HSup , OCount) ->
    HiddenNeuronPids = get_hidden_neurons(HCount, HSup, []),
    [e_ann_hidden_neuron:init_weights(Pid, OCount) || Pid <- HiddenNeuronPids],
    HiddenNeuronPids.

create_input_layer(TrainingData, ICount, ISup, HCount) ->
    InputNeuronPids = get_input_neurons(ICount, ISup, TrainingData, []),
    [e_ann_input_neuron:init_weights(Pid, HCount) || Pid <- InputNeuronPids],
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

get_input_neurons(0, _, [], Acc) ->
    Acc;
get_input_neurons(ICount, ISup, Inputs, Acc) ->
    {ok, Pid} = e_ann_input_neuron_sup:add_child(ISup, hd(Inputs)),
    NewCount = ICount - 1,
    Acc2 = [Pid | Acc],
    get_input_neurons(NewCount, ISup, tl(Inputs), Acc2).

get_hidden_neurons(0, _, Acc) ->
    Acc;
get_hidden_neurons(HCount, HSup, Acc) ->
    {ok, Pid} = e_ann_hidden_neuron_sup:add_child(HSup),
    NewCount = HCount - 1,
    Acc2 = [Pid | Acc],
    get_hidden_neurons(NewCount, HSup, Acc2).

get_output_neurons(0, _, [], Acc) ->
    Acc;
get_output_neurons(OCount, OSup, Ideal, Acc) ->
    {ok, Pid} = e_ann_output_neuron_sup:add_child(OSup, hd(Ideal)),
    NewCount = OCount - 1,
    Acc2 = [Pid | Acc],
    get_output_neurons(NewCount, OSup, tl(Ideal), Acc2).
