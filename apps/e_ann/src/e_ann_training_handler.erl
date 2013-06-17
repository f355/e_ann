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

-compile([export_all]).

train(Inputs, Ideal, Architecture, ErrorRate) ->
    ok.

%% Example of architecture [{input_neurons,2,true}, {hidden_layers,1, true},
%%                          {hidden_layer_neurons, 2},{output_neurons,1,true}]
read_architecture_and_spawn_neurons(Architecture) ->
    [{bias_sup,BSup}, {hidden_sup, HSup},
     {output_sup, OSup}, {input_sup, ISup}] = get_neuron_sup_pids(e_ann_sup),
    [{_,INeuronCount, IBias}, {_,LayerCount,HBias},
     {_, HNeuronCount}, {_,ONeuronCount,OBias}] = Architecture,
    InputLayer = 

get_neuron_sup_pids(Sup) when is_atom(Sup) ->
    [{_, BSup, _, _}, {_, HSup, _, _},
     {_, OSup, _, _}, {_, ISup, _, _}] = supervisor:which_children(Sup),
    [{bias_sup, BSup},{hidden_sup, HSup},{output_sup, OSup},{input_sup, ISup}].


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Internal Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input_layer(ICount, ISup, IBias, Inputs) ->
    ok.

get_input_neurons(0, _, [], Acc) ->
    Acc;
get_input_neurons(ICount, ISup, Inputs, Acc) ->
    {ok, Pid} = e_ann_input_neuron_sup:add_child(ISup, hd(Inputs)),
    NewCount = ICount - 1,
    Acc2 = [Pid | Acc],
    get_input_neurons(NewCount, ISup, tl(Inputs), Acc2).
