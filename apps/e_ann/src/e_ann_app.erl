%%%-------------------------------------------------------------------
%%% @doc
%%% The application callback module for e_ann.
%%% @end
%%%
%%%-------------------------------------------------------------------
-module(e_ann_app).

-behaviour(application).

%% Application callbacks
-export([start/2, stop/1]).

%% ===================================================================
%% Application callbacks
%% ===================================================================
start(normal, no_arg) ->
    e_ann_sup:start_link().

%%--------------------------------------------------------------------
%% Function: stop(Data) -> ok.
%% @doc
%%   Stops the application.
%% @end
%%--------------------------------------------------------------------
-spec stop(Data::_) -> ok.
%%--------------------------------------------------------------------
stop(_) ->
    ok.
