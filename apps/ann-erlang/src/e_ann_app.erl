-module(e_ann_app).

-behaviour(application).

%% Application callbacks
-export([start/2, stop/1]).

%% ===================================================================
%% Application callbacks
%% ===================================================================

start(_, _) ->
    application:start(log4erl),
    log4erl:conf("/home/cantheman/Project/ann/apps/ann-erlang/priv/log4erl.conf"),
    e_ann_sup:start_link().

stop(_State) ->
    ok.
