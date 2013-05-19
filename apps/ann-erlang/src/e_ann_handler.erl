%%%-------------------------------------------------------------------
%%% @author cantheman <can@campanja.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% This module reads all the inputs and divides them among the pool
%%% processes. Should more processes be needed it spawns more before
%%% it starts training on input data.
%%% @end
%%% Created :  19 May 2013 by cantheman <can@campanja.com>
%%%-------------------------------------------------------------------
-module(e_ann_handler).

train(Input, Ideal, Architecture, ErrorRate) ->
    M = length(Input).
    
