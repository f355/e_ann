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
    [{_,BSup},{_,HSup},{_,OSup},{_,ISup}] = e_ann_training_handler:get_neuron_sup_pids(),
    Ilayer = e_ann_training_handler:input_layer([1.0,0.0],2,ISup,BSup,false),
    Hlayer = e_ann_training_handler:hidden_layer(2,HSup,BSup,false),
    Olayer = e_ann_training_handler:output_layer([1.0],1,OSup,BSup,false),
    I1 = hd(Ilayer),
    [I2] = tl(Ilayer),
    H1 = hd(Hlayer),
    [H2] = tl(Hlayer),
    O1 = hd(Olayer),
    [I1,I2, H1, H2, O1].

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Internal Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output_layer(Ideal, OCount, OSup, BSup, BiasConfig) ->
    OutputNeuronPids = get_output_neurons(OCount, OSup, Ideal, []),
    check_for_bias(BiasConfig, BSup, OutputNeuronPids).

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

get_output_neurons(0, _, [], Acc) ->
    Acc;
get_output_neurons(OCount, OSup, Ideal, Acc) ->
    {ok, Pid} = e_ann_output_neuron_sup:add_child(OSup, hd(Ideal)),
    NewCount = OCount - 1,
    Acc2 = [Pid | Acc],
    get_output_neurons(NewCount, OSup, tl(Ideal), Acc2).
