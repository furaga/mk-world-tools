GAMES = mk8dx-race mk8dx-battle mkworld-race mkworld-survival mkworld-battle
ARGS ?=  # ARGSが未定義の場合、空文字列をデフォルト値として設定

define run_recorder
$(1):
	uv run python ./scripts/auto_recorder.py --out_csv_path data/record/$(1).csv --game $(1) $(ARGS)
endef

$(foreach game,$(GAMES),$(eval $(call run_recorder,$(game))))
