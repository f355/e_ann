%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module has all the utility functions for creating a neural
%%% network.
%%% @end
%%% Created :  19 June 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_network).

-export([get_sup_pids/0, create_layer_with_random_weights/5,
         spawn_neurons/4, spawn_bias_neuron/4, set_inputs/2]).

-define(MAINSUPERVISOR, e_ann_sup).

get_sup_pids() ->
    [{_, IBSup, _, _}, {_, HBSup,_ ,_}, {_, HSup, _, _},
     {_, OSup, _, _}, {_, ISup, _, _}] =
        supervisor:which_children(?MAINSUPERVISOR),
    [{input_bias_sup, IBSup},{hidden_bias_sup, HBSup}, {hidden_sup, HSup},
     {output_sup, OSup},{input_sup, ISup}].

create_layer_with_random_weights(Count, Sup, SF, F, HCount) ->
    Neurons = spawn_neurons(Count, Sup, SF, []),
    [ F(Pid, HCount) || Pid <- Neurons ],
    Neurons.

spawn_neurons(0, _, _, Acc) ->
    Acc;
spawn_neurons(Count, Sup, F, Acc) ->
    {ok, Pid} = F(Sup),
    NewCount = Count - 1,
    NewAcc = [Pid | Acc],
    spawn_neurons(NewCount, Sup, F, NewAcc).

spawn_bias_neuron(Sup, WeightCount, SF, F) ->
    {ok, Pid} = SF(Sup),
    F(Pid, WeightCount),
    Pid.

set_inputs([], []) ->
    ok;
set_inputs(Layer, Inputs) ->
    e_ann_input_neuron:add_input(hd(Layer), hd(Inputs)),
    set_inputs(tl(Layer), tl(Inputs)).


