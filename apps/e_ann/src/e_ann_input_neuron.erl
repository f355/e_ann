%%%-------------------------------------------------------------------
%%% @author cantheman <can@campanja.com>
%%% @copyright (C) 2013, cantheman
%%% @doc
%%%
%%% @end
%%% Created :  9 Mar 2013 by cantheman <can@campanja.com>
%%%-------------------------------------------------------------------
-module(e_ann_input_neuron).

-behaviour(gen_server).

%% API
-export([start_link/0, feed_forward/4]).

%% gen_server callbacks
-export([init/1, handle_call/3, handle_cast/2, handle_info/2,
	 terminate/2, code_change/3]).

-define(SERVER, ?MODULE).

-record(state, {input, weight, output}).

%%%===================================================================
%%% API
%%%===================================================================

start_link() ->
    gen_server:start_link(?MODULE, [], []).

feed_forward(Pid, Input, Weight, Output) ->
    gen_server:call(Pid, {feed_forward, Input, Weight, Output}).

%%%===================================================================
%%% gen_server callbacks
%%%===================================================================

init([]) ->
    Weight = random:uniform(),
    log4erl:log(info, "Starting ~p Input neuron with weight~p~n",
		[self(), Weight]),
    State = #state{weight=Weight},
    {ok, State}.

handle_call({feed_forward, Input, Weight, Output}, _From, State) ->
    io:format("input~p weight~p output~p~n",[Input,Weight,Output]),
    {reply, ok, State}.

handle_cast(_Msg, State) ->
    {noreply, State}.

handle_info(_Info, State) ->
    {noreply, State}.

terminate(_Reason, _State) ->
    ok.

code_change(_OldVsn, State, _Extra) ->
    {ok, State}.

%%%===================================================================
%%% Internal functions
%%%===================================================================

