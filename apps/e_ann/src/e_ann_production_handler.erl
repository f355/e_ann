%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module starts a production network with trained weights that
%%% can be used for e.g. predictions and pattern recognition.
%%% @end
%%% Created :  23 July 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_production_handler).

init_network(Weights) ->
    [{_,IBSup},{_, HBSup},{_,HSup},{_,OSup},{_,ISup}] =
        e_ann_training_handler:get_neuron_sup_pids(),
    ok.
