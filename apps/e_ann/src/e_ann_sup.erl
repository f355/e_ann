%%%-------------------------------------------------------------------
%%% @author cantheman <java10cana@gmail.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%%
%%% @end
%%% Created : 18 June 2013 by cantheman <java10cana@gmail.com>
%%%-------------------------------------------------------------------
-module(e_ann_sup).

-behaviour(supervisor).

%% API
-export([start_link/0]).

%% Supervisor callbacks
-export([init/1]).

%% ===================================================================
%% API functions
%% ===================================================================

start_link() ->
    supervisor:start_link({local, ?MODULE}, ?MODULE, []).

%% ===================================================================
%% Supervisor callbacks
%% ===================================================================

init(_Args) ->
    log4erl:info("Starting e_ann supervisor with pid:(~p)~n", [self()]),
    RestartStrategy = {one_for_one, 5, 10},
    Supervisors = [child(e_ann_input_neuron_sup),
                   child(e_ann_output_neuron_sup),
                   child(e_ann_hidden_neuron_sup),
                   child(e_ann_hidden_bias_neuron_sup),
                   child(e_ann_input_bias_neuron_sup)],
    {ok, {RestartStrategy, Supervisors}}.

child(Module) ->
    {Module, {Module, start_link, []}, permanent, 2000, supervisor, [Module]}.
