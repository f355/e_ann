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

%train(TrainingData, Architecture, ErrorRate)
%  when length(Inputs) =:= length(Ideal) ->
%    ok.

%% Example of architecture [{input_neurons,2,true}, {hidden_layers,1, true},
%%                          {hidden_layer_neurons, 2},{output_neurons,1,true}]
%% read_architecture_and_spawn_neurons(Architecture) ->
%%     [{bias_sup,BSup}, {hidden_sup, HSup},
%%      {output_sup, OSup}, {input_sup, ISup}] = get_neuron_sup_pids(e_ann_sup),
%%     [{_, INeuronCount, IBias}, {_, LayerCount, HBias},
%%      {_, HNeuronCount}, {_,ONeuronCount,OBias}] = Architecture,
%%     ok.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Internal Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hidden_layer(HCount, HSup, BSup, BiasConfig) ->
    HiddenNeuronPids = get_hidden_neurons(HCount, HSup, []),
    check_for_bias(BiasConfig, BSup, HiddenNeuronPids).

input_layer(TrainingData, ICount, ISup, BSup, BiasConfig) ->
    InputNeuronPids = get_input_neurons(ICount, ISup, TrainingData, []),
    check_for_bias(BiasConfig, BSup, InputNeuronPids).

bias(Config, BSup) ->
    case Config of
        true ->
            {ok, Pid} = e_ann_bias_neuron_sup:add_child(BSup),
            Pid;
        false ->
            []
    end.

get_neuron_sup_pids() ->
    [{_, BSup, _, _}, {_, HSup, _, _},
     {_, OSup, _, _}, {_, ISup, _, _}] =
        supervisor:which_children(?MAINSUPERVISOR),
    [{bias_sup, BSup},{hidden_sup, HSup},{output_sup, OSup},{input_sup, ISup}].


check_for_bias(BiasConfig, BSup, Neurons) ->
    Bias = bias(BiasConfig, BSup),
    case Bias of
        [] ->
            Neurons;
        Bias ->
            [Bias | Neurons]
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

