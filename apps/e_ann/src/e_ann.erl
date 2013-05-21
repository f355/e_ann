%%%-------------------------------------------------------------------
%%% @doc
%%% The application façade module for e_ann.
%%% @end
%%%
%%%-------------------------------------------------------------------
-module(e_ann).

%% Management API
-export([start/0]).

start() ->
    application:start(log4erl),
    setup_logging(),
    application:start(?MODULE).

%% ===================================================================
%% Internal functions.
%% ===================================================================

setup_logging() ->
    application:load(?MODULE),
    Log4ErlConf = filename:join([code:priv_dir(?MODULE), "log4erl.conf"]),
    log4erl:conf(Log4ErlConf),
    log4erl:error_logger_handler().
