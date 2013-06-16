%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module reads the architecture config and lets the supervisors spawn
%%% child processes accordingly.
%%% @end
%%% Created :  19 May 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_training_handler).

-compile([export_all]).

%%train(Input, Ideal, Architecture, ErrorRate) ->
%%    ok.

get_neuron_sup_pids(Sup) ->
    [{_, BSup, _, _}, {_, HSup, _, _},
     {_, OSup, _, _}, {_, ISup, _, _}] = supervisor:which_children(Sup),
    [{bias_sup, BSup},{hidden_sup, HSup},{output_sup, OSup},{input_sup, ISup}].


