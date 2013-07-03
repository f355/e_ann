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
    HCount = 2,
    OCount = 1,
    Ilayer = e_ann_training_handler:input_layer([1.0,0.0],2, ISup, HCount),
    Hlayer = e_ann_training_handler:hidden_layer(HCount ,HSup , OCount),
    Olayer = e_ann_training_handler:output_layer([1.0], OCount, OSup),
    IBias = input_bias(true, IBSup, 2),
    HBias = hidden_bias(true, HBSup, 1),
    input_layer_activation(Ilayer, Hlayer),
    hidden_layer_activation(Hlayer, Olayer),
    e_ann_output_neuron:activate_neuron(hd(Olayer)).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Internal Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input_layer_activation(Ilayer, Hlayer) ->
    [ e_ann_input_neuron:calculate_output(Neuron, Hlayer) || Neuron <- Ilayer ].

hidden_layer_activation(Hlayer, Olayer) ->
    [ e_ann_hidden_neuron:activate_neuron(Neuron) || Neuron <- Hlayer ],
    [ e_ann_hidden_neuron:calculate_output(Neuron,Olayer) || Neuron <- Hlayer ].

output_layer(Ideal, OCount, OSup) ->
    get_output_neurons(OCount, OSup, Ideal, []).

hidden_layer(HCount, HSup , OCount) ->
    HiddenNeuronPids = get_hidden_neurons(HCount, HSup, []),
    [e_ann_hidden_neuron:init_weights(Pid, OCount) || Pid <- HiddenNeuronPids].

input_layer(TrainingData, ICount, ISup, HCount) ->
    InputNeuronPids = get_input_neurons(ICount, ISup, TrainingData, []),
    [e_ann_input_neuron:init_weights(Pid, HCount) || Pid <- InputNeuronPids].

get_neuron_sup_pids() ->
    [{_, IBSup, _, _}, {_, HBSup,_ ,_}, {_, HSup, _, _},
     {_, OSup, _, _}, {_, ISup, _, _}] =
        supervisor:which_children(?MAINSUPERVISOR),
    [{input_bias_sup, IBSup},{hidden_bias_sup, HBSup}, {hidden_sup, HSup},
     {output_sup, OSup},{input_sup, ISup}].


input_bias(Config, Sup, Count) ->
    case Config of
        true ->
            {ok, Pid} = e_ann_input_bias_neuron_sup:add_child(Sup),
            e_ann_input_bias_neuron:init_weights(Pid, Count),
            Pid;
        false ->
            []
    end.

hidden_bias(Config, Sup, Count) ->
    case Config of
        true ->
            {ok, Pid} = e_ann_hidden_bias_neuron_sup:add_child(Sup),
            e_ann_hidden_bias_neuron:init_weights(Pid, Count),
            Pid;
        false ->
            []
    end.

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
