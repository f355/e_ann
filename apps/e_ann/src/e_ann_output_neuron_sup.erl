%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%% Output neuron supervisor who dynamically adds output neurons.
%%% @end
%%% Created : 14 June 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_output_neuron_sup).

-behaviour(supervisor).

%% API
-export([start_link/0, add_child/2]).

%% Supervisor callbacks
-export([init/1]).

%% ===================================================================
%% API functions
%% ===================================================================

start_link() ->
    supervisor:start_link(?MODULE, []).

%% ===================================================================
%% Supervisor callbacks
%% ===================================================================

init([]) ->
    log4erl:log(info, "Starting e_ann_output_neuron supervisor (~p)~n",
                [self()]),
    RestartStrategy = {simple_one_for_one, 0, 1},
    Children = [child(e_ann_output_neuron, [])],
    {ok, {RestartStrategy, Children}}.

add_child(Sup, Ideal) ->
    supervisor:start_child(Sup, [Ideal]).

child(Module, Ideal) ->
    {Module, {Module, start_link, Ideal}, temporary, 2000, worker, [Module]}.
