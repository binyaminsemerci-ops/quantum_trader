# Project Status Tree (updated 2025-09-29 04:48:50 UTC)

Legend: ✅ done | ⚪ next | ❌ pending

## Tree
✅ .
|-- ✅ .githooks/
|   \-- ✅ pre-commit
|-- ✅ .github/
|   |-- ✅ auto-fix/
|   |   \-- ✅ auto_fix.sh
|   |-- ✅ workflows/
|   |   |-- ✅ auto-approve.yml
|   |   |-- ✅ auto-fix-and-push.yml
|   |   |-- ✅ auto-fix-on-failed-workflow.yml
|   |   |-- ✅ auto_label_on_comment.yml
|   |   |-- ✅ ci-integration.yml
|   |   |-- ✅ ci-mypy.yml
|   |   |-- ✅ ci.yml
|   |   |-- ✅ daily-frontend-audit.yml
|   |   |-- ✅ deployment-build.yml
|   |   |-- ✅ dev-tests.yml
|   |   |-- ✅ docker-publish-3.yml
|   |   |-- ✅ frontend-ci.yml
|   |   |-- ✅ frontend-docker-test.yml
|   |   |-- ✅ frontend.yml
|   |   |-- ✅ nightly-dev-check.yml
|   |   |-- ✅ precommit.yml
|   |   |-- ✅ stress-tests.yml
|   |   \-- ✅ tests.yml
|   |-- ✅ GHCR_PAT.md
|   \-- ✅ pr-rerun-trigger.txt
|-- ✅ ai_engine/
|   |-- ✅ agents/
|   |   \-- ✅ xgb_agent.py
|   |-- ✅ backend/
|   |   |-- ✅ routes/
|   |   |   |-- ✅ chart.py
|   |   |   |-- ✅ stats.py
|   |   |   \-- ✅ trades.py
|   |   |-- ✅ utils/
|   |   |   |-- ✅ binance_client.py
|   |   |   |-- ✅ failsafe.py
|   |   |   |-- ✅ risk.py
|   |   |   \-- ✅ trade_logger.py
|   |   \-- ✅ main.py
|   |-- ✅ models/
|   |   |-- ✅ scaler.pkl
|   |   |-- ✅ xgb_model.json
|   |   \-- ✅ xgb_model.pkl
|   |-- ✅ __init__.py
|   |-- ✅ feature_engineer.py
|   \-- ✅ train_and_save.py
|-- ✅ artifacts/
|   \-- ✅ stress/
|       |-- ✅ experiments/
|       |   |-- ✅ node-node-18-bullseye-slim__deps-baseline/
|       |   |-- ✅ node-node-18-bullseye-slim__deps-playwright-1-39-0/
|       |   |-- ✅ node-node-20-bullseye-slim__deps-baseline/
|       |   |-- ✅ node-node-20-bullseye-slim__deps-playwright-1-39-0/
|       |   |-- ✅ node20-baseline/
|       |   |   |-- ✅ aggregated.json
|       |   |   \-- ✅ iter_0001.json
|       |   \-- ✅ index.json
|       |-- ✅ aggregated.json
|       \-- ✅ iter_0001.json
|-- ✅ artifacts_17932681103/
|   |-- ✅ pip-list-after.txt
|   \-- ✅ pip-list-before.txt
|-- ✅ backend/
|   |-- ✅ .venv/
|   |   |-- ✅ Include/
|   |   |   \-- ✅ site/
|   |   |       \-- ✅ python3.12/
|   |   |           \-- ✅ greenlet/
|   |   |               \-- ✅ greenlet.h
|   |   |-- ✅ Lib/
|   |   |   \-- ✅ site-packages/
|   |   |       |-- ✅ aiohappyeyeballs/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _staggered.py
|   |   |       |   |-- ✅ impl.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ types.py
|   |   |       |   \-- ✅ utils.py
|   |   |       |-- ✅ aiohappyeyeballs-2.6.1.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ aiohttp/
|   |   |       |   |-- ✅ .hash/
|   |   |       |   |   |-- ✅ _cparser.pxd.hash
|   |   |       |   |   |-- ✅ _find_header.pxd.hash
|   |   |       |   |   |-- ✅ _http_parser.pyx.hash
|   |   |       |   |   |-- ✅ _http_writer.pyx.hash
|   |   |       |   |   \-- ✅ hdrs.py.hash
|   |   |       |   |-- ✅ _websocket/
|   |   |       |   |   |-- ✅ .hash/
|   |   |       |   |   |   |-- ✅ mask.pxd.hash
|   |   |       |   |   |   |-- ✅ mask.pyx.hash
|   |   |       |   |   |   \-- ✅ reader_c.pxd.hash
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ helpers.py
|   |   |       |   |   |-- ✅ mask.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ mask.pxd
|   |   |       |   |   |-- ✅ mask.pyx
|   |   |       |   |   |-- ✅ models.py
|   |   |       |   |   |-- ✅ reader.py
|   |   |       |   |   |-- ✅ reader_c.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ reader_c.pxd
|   |   |       |   |   |-- ✅ reader_c.py
|   |   |       |   |   |-- ✅ reader_py.py
|   |   |       |   |   \-- ✅ writer.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _cookie_helpers.py
|   |   |       |   |-- ✅ _cparser.pxd
|   |   |       |   |-- ✅ _find_header.pxd
|   |   |       |   |-- ✅ _headers.pxi
|   |   |       |   |-- ✅ _http_parser.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _http_parser.pyx
|   |   |       |   |-- ✅ _http_writer.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _http_writer.pyx
|   |   |       |   |-- ✅ abc.py
|   |   |       |   |-- ✅ base_protocol.py
|   |   |       |   |-- ✅ client.py
|   |   |       |   |-- ✅ client_exceptions.py
|   |   |       |   |-- ✅ client_middleware_digest_auth.py
|   |   |       |   |-- ✅ client_middlewares.py
|   |   |       |   |-- ✅ client_proto.py
|   |   |       |   |-- ✅ client_reqrep.py
|   |   |       |   |-- ✅ client_ws.py
|   |   |       |   |-- ✅ compression_utils.py
|   |   |       |   |-- ✅ connector.py
|   |   |       |   |-- ✅ cookiejar.py
|   |   |       |   |-- ✅ formdata.py
|   |   |       |   |-- ✅ hdrs.py
|   |   |       |   |-- ✅ helpers.py
|   |   |       |   |-- ✅ http.py
|   |   |       |   |-- ✅ http_exceptions.py
|   |   |       |   |-- ✅ http_parser.py
|   |   |       |   |-- ✅ http_websocket.py
|   |   |       |   |-- ✅ http_writer.py
|   |   |       |   |-- ✅ log.py
|   |   |       |   |-- ✅ multipart.py
|   |   |       |   |-- ✅ payload.py
|   |   |       |   |-- ✅ payload_streamer.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ pytest_plugin.py
|   |   |       |   |-- ✅ resolver.py
|   |   |       |   |-- ✅ streams.py
|   |   |       |   |-- ✅ tcp_helpers.py
|   |   |       |   |-- ✅ test_utils.py
|   |   |       |   |-- ✅ tracing.py
|   |   |       |   |-- ✅ typedefs.py
|   |   |       |   |-- ✅ web.py
|   |   |       |   |-- ✅ web_app.py
|   |   |       |   |-- ✅ web_exceptions.py
|   |   |       |   |-- ✅ web_fileresponse.py
|   |   |       |   |-- ✅ web_log.py
|   |   |       |   |-- ✅ web_middlewares.py
|   |   |       |   |-- ✅ web_protocol.py
|   |   |       |   |-- ✅ web_request.py
|   |   |       |   |-- ✅ web_response.py
|   |   |       |   |-- ✅ web_routedef.py
|   |   |       |   |-- ✅ web_runner.py
|   |   |       |   |-- ✅ web_server.py
|   |   |       |   |-- ✅ web_urldispatcher.py
|   |   |       |   |-- ✅ web_ws.py
|   |   |       |   \-- ✅ worker.py
|   |   |       |-- ✅ aiohttp-3.12.15.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ vendor/
|   |   |       |   |   |   \-- ✅ llhttp/
|   |   |       |   |   |       \-- ✅ LICENSE
|   |   |       |   |   \-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ aiosignal/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ aiosignal-1.4.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ annotated_types/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   \-- ✅ test_cases.py
|   |   |       |-- ✅ annotated_types-0.7.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ anyio/
|   |   |       |   |-- ✅ _backends/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _asyncio.py
|   |   |       |   |   \-- ✅ _trio.py
|   |   |       |   |-- ✅ _core/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _asyncio_selector_thread.py
|   |   |       |   |   |-- ✅ _contextmanagers.py
|   |   |       |   |   |-- ✅ _eventloop.py
|   |   |       |   |   |-- ✅ _exceptions.py
|   |   |       |   |   |-- ✅ _fileio.py
|   |   |       |   |   |-- ✅ _resources.py
|   |   |       |   |   |-- ✅ _signals.py
|   |   |       |   |   |-- ✅ _sockets.py
|   |   |       |   |   |-- ✅ _streams.py
|   |   |       |   |   |-- ✅ _subprocesses.py
|   |   |       |   |   |-- ✅ _synchronization.py
|   |   |       |   |   |-- ✅ _tasks.py
|   |   |       |   |   |-- ✅ _tempfile.py
|   |   |       |   |   |-- ✅ _testing.py
|   |   |       |   |   \-- ✅ _typedattr.py
|   |   |       |   |-- ✅ abc/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _eventloop.py
|   |   |       |   |   |-- ✅ _resources.py
|   |   |       |   |   |-- ✅ _sockets.py
|   |   |       |   |   |-- ✅ _streams.py
|   |   |       |   |   |-- ✅ _subprocesses.py
|   |   |       |   |   |-- ✅ _tasks.py
|   |   |       |   |   \-- ✅ _testing.py
|   |   |       |   |-- ✅ streams/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ buffered.py
|   |   |       |   |   |-- ✅ file.py
|   |   |       |   |   |-- ✅ memory.py
|   |   |       |   |   |-- ✅ stapled.py
|   |   |       |   |   |-- ✅ text.py
|   |   |       |   |   \-- ✅ tls.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ from_thread.py
|   |   |       |   |-- ✅ lowlevel.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ pytest_plugin.py
|   |   |       |   |-- ✅ to_interpreter.py
|   |   |       |   |-- ✅ to_process.py
|   |   |       |   \-- ✅ to_thread.py
|   |   |       |-- ✅ anyio-4.11.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ attr/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __init__.pyi
|   |   |       |   |-- ✅ _cmp.py
|   |   |       |   |-- ✅ _cmp.pyi
|   |   |       |   |-- ✅ _compat.py
|   |   |       |   |-- ✅ _config.py
|   |   |       |   |-- ✅ _funcs.py
|   |   |       |   |-- ✅ _make.py
|   |   |       |   |-- ✅ _next_gen.py
|   |   |       |   |-- ✅ _typing_compat.pyi
|   |   |       |   |-- ✅ _version_info.py
|   |   |       |   |-- ✅ _version_info.pyi
|   |   |       |   |-- ✅ converters.py
|   |   |       |   |-- ✅ converters.pyi
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ exceptions.pyi
|   |   |       |   |-- ✅ filters.py
|   |   |       |   |-- ✅ filters.pyi
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ setters.py
|   |   |       |   |-- ✅ setters.pyi
|   |   |       |   |-- ✅ validators.py
|   |   |       |   \-- ✅ validators.pyi
|   |   |       |-- ✅ attrs/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __init__.pyi
|   |   |       |   |-- ✅ converters.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ filters.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ setters.py
|   |   |       |   \-- ✅ validators.py
|   |   |       |-- ✅ attrs-25.3.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ binance/
|   |   |       |   |-- ✅ ccxt/
|   |   |       |   |   |-- ✅ abstract/
|   |   |       |   |   |   \-- ✅ binance.py
|   |   |       |   |   |-- ✅ async_support/
|   |   |       |   |   |   |-- ✅ base/
|   |   |       |   |   |   |   |-- ✅ ws/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ cache.py
|   |   |       |   |   |   |   |   |-- ✅ client.py
|   |   |       |   |   |   |   |   |-- ✅ functions.py
|   |   |       |   |   |   |   |   |-- ✅ future.py
|   |   |       |   |   |   |   |   |-- ✅ order_book.py
|   |   |       |   |   |   |   |   \-- ✅ order_book_side.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ exchange.py
|   |   |       |   |   |   |   \-- ✅ throttler.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ binance.py
|   |   |       |   |   |-- ✅ base/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ decimal_to_precision.py
|   |   |       |   |   |   |-- ✅ errors.py
|   |   |       |   |   |   |-- ✅ exchange.py
|   |   |       |   |   |   |-- ✅ precise.py
|   |   |       |   |   |   \-- ✅ types.py
|   |   |       |   |   |-- ✅ pro/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ binance.py
|   |   |       |   |   |-- ✅ static_dependencies/
|   |   |       |   |   |   |-- ✅ ecdsa/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _version.py
|   |   |       |   |   |   |   |-- ✅ curves.py
|   |   |       |   |   |   |   |-- ✅ der.py
|   |   |       |   |   |   |   |-- ✅ ecdsa.py
|   |   |       |   |   |   |   |-- ✅ ellipticcurve.py
|   |   |       |   |   |   |   |-- ✅ keys.py
|   |   |       |   |   |   |   |-- ✅ numbertheory.py
|   |   |       |   |   |   |   |-- ✅ rfc6979.py
|   |   |       |   |   |   |   \-- ✅ util.py
|   |   |       |   |   |   |-- ✅ ethereum/
|   |   |       |   |   |   |   |-- ✅ abi/
|   |   |       |   |   |   |   |   |-- ✅ tools/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   \-- ✅ _strategies.py
|   |   |       |   |   |   |   |   |-- ✅ utils/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ numeric.py
|   |   |       |   |   |   |   |   |   |-- ✅ padding.py
|   |   |       |   |   |   |   |   |   \-- ✅ string.py
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ abi.py
|   |   |       |   |   |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |   |   |-- ✅ codec.py
|   |   |       |   |   |   |   |   |-- ✅ constants.py
|   |   |       |   |   |   |   |   |-- ✅ decoding.py
|   |   |       |   |   |   |   |   |-- ✅ encoding.py
|   |   |       |   |   |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |   |   |-- ✅ grammar.py
|   |   |       |   |   |   |   |   |-- ✅ packed.py
|   |   |       |   |   |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   |   |   \-- ✅ registry.py
|   |   |       |   |   |   |   |-- ✅ account/
|   |   |       |   |   |   |   |   |-- ✅ encode_typed_data/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ encoding_and_hashing.py
|   |   |       |   |   |   |   |   |   \-- ✅ helpers.py
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ messages.py
|   |   |       |   |   |   |   |   \-- ✅ py.typed
|   |   |       |   |   |   |   |-- ✅ hexbytes/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ _utils.py
|   |   |       |   |   |   |   |   |-- ✅ main.py
|   |   |       |   |   |   |   |   \-- ✅ py.typed
|   |   |       |   |   |   |   |-- ✅ typing/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ abi.py
|   |   |       |   |   |   |   |   |-- ✅ bls.py
|   |   |       |   |   |   |   |   |-- ✅ discovery.py
|   |   |       |   |   |   |   |   |-- ✅ encoding.py
|   |   |       |   |   |   |   |   |-- ✅ enums.py
|   |   |       |   |   |   |   |   |-- ✅ ethpm.py
|   |   |       |   |   |   |   |   |-- ✅ evm.py
|   |   |       |   |   |   |   |   |-- ✅ networks.py
|   |   |       |   |   |   |   |   \-- ✅ py.typed
|   |   |       |   |   |   |   |-- ✅ utils/
|   |   |       |   |   |   |   |   |-- ✅ curried/
|   |   |       |   |   |   |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ typing/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   \-- ✅ misc.py
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ abi.py
|   |   |       |   |   |   |   |   |-- ✅ address.py
|   |   |       |   |   |   |   |   |-- ✅ applicators.py
|   |   |       |   |   |   |   |   |-- ✅ conversions.py
|   |   |       |   |   |   |   |   |-- ✅ currency.py
|   |   |       |   |   |   |   |   |-- ✅ debug.py
|   |   |       |   |   |   |   |   |-- ✅ decorators.py
|   |   |       |   |   |   |   |   |-- ✅ encoding.py
|   |   |       |   |   |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |   |   |-- ✅ functional.py
|   |   |       |   |   |   |   |   |-- ✅ hexadecimal.py
|   |   |       |   |   |   |   |   |-- ✅ humanize.py
|   |   |       |   |   |   |   |   |-- ✅ logging.py
|   |   |       |   |   |   |   |   |-- ✅ module_loading.py
|   |   |       |   |   |   |   |   |-- ✅ numeric.py
|   |   |       |   |   |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   |   |   |-- ✅ toolz.py
|   |   |       |   |   |   |   |   |-- ✅ types.py
|   |   |       |   |   |   |   |   \-- ✅ units.py
|   |   |       |   |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ keccak/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ keccak.py
|   |   |       |   |   |   |-- ✅ lark/
|   |   |       |   |   |   |   |-- ✅ __pyinstaller/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   \-- ✅ hook-lark.py
|   |   |       |   |   |   |   |-- ✅ grammars/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ common.lark
|   |   |       |   |   |   |   |   |-- ✅ lark.lark
|   |   |       |   |   |   |   |   |-- ✅ python.lark
|   |   |       |   |   |   |   |   \-- ✅ unicode.lark
|   |   |       |   |   |   |   |-- ✅ parsers/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ cyk.py
|   |   |       |   |   |   |   |   |-- ✅ earley.py
|   |   |       |   |   |   |   |   |-- ✅ earley_common.py
|   |   |       |   |   |   |   |   |-- ✅ earley_forest.py
|   |   |       |   |   |   |   |   |-- ✅ grammar_analysis.py
|   |   |       |   |   |   |   |   |-- ✅ lalr_analysis.py
|   |   |       |   |   |   |   |   |-- ✅ lalr_interactive_parser.py
|   |   |       |   |   |   |   |   |-- ✅ lalr_parser.py
|   |   |       |   |   |   |   |   |-- ✅ lalr_parser_state.py
|   |   |       |   |   |   |   |   \-- ✅ xearley.py
|   |   |       |   |   |   |   |-- ✅ tools/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ nearley.py
|   |   |       |   |   |   |   |   |-- ✅ serialize.py
|   |   |       |   |   |   |   |   \-- ✅ standalone.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ ast_utils.py
|   |   |       |   |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |   |-- ✅ grammar.py
|   |   |       |   |   |   |   |-- ✅ indenter.py
|   |   |       |   |   |   |   |-- ✅ lark.py
|   |   |       |   |   |   |   |-- ✅ lexer.py
|   |   |       |   |   |   |   |-- ✅ load_grammar.py
|   |   |       |   |   |   |   |-- ✅ parse_tree_builder.py
|   |   |       |   |   |   |   |-- ✅ parser_frontends.py
|   |   |       |   |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   |   |-- ✅ reconstruct.py
|   |   |       |   |   |   |   |-- ✅ tree.py
|   |   |       |   |   |   |   |-- ✅ tree_matcher.py
|   |   |       |   |   |   |   |-- ✅ tree_templates.py
|   |   |       |   |   |   |   |-- ✅ utils.py
|   |   |       |   |   |   |   \-- ✅ visitors.py
|   |   |       |   |   |   |-- ✅ marshmallow/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |   |-- ✅ class_registry.py
|   |   |       |   |   |   |   |-- ✅ decorators.py
|   |   |       |   |   |   |   |-- ✅ error_store.py
|   |   |       |   |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |   |-- ✅ fields.py
|   |   |       |   |   |   |   |-- ✅ orderedset.py
|   |   |       |   |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   |   |-- ✅ schema.py
|   |   |       |   |   |   |   |-- ✅ types.py
|   |   |       |   |   |   |   |-- ✅ utils.py
|   |   |       |   |   |   |   |-- ✅ validate.py
|   |   |       |   |   |   |   \-- ✅ warnings.py
|   |   |       |   |   |   |-- ✅ marshmallow_dataclass/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ collection_field.py
|   |   |       |   |   |   |   |-- ✅ lazy_class_attribute.py
|   |   |       |   |   |   |   |-- ✅ mypy.py
|   |   |       |   |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   |   |-- ✅ typing.py
|   |   |       |   |   |   |   \-- ✅ union_field.py
|   |   |       |   |   |   |-- ✅ marshmallow_oneofschema/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ one_of_schema.py
|   |   |       |   |   |   |   \-- ✅ py.typed
|   |   |       |   |   |   |-- ✅ msgpack/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _cmsgpack.pyx
|   |   |       |   |   |   |   |-- ✅ _packer.pyx
|   |   |       |   |   |   |   |-- ✅ _unpacker.pyx
|   |   |       |   |   |   |   |-- ✅ buff_converter.h
|   |   |       |   |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |   |-- ✅ ext.py
|   |   |       |   |   |   |   |-- ✅ fallback.py
|   |   |       |   |   |   |   |-- ✅ pack.h
|   |   |       |   |   |   |   |-- ✅ pack_template.h
|   |   |       |   |   |   |   |-- ✅ sysdep.h
|   |   |       |   |   |   |   |-- ✅ unpack.h
|   |   |       |   |   |   |   |-- ✅ unpack_define.h
|   |   |       |   |   |   |   \-- ✅ unpack_template.h
|   |   |       |   |   |   |-- ✅ parsimonious/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |   |-- ✅ expressions.py
|   |   |       |   |   |   |   |-- ✅ grammar.py
|   |   |       |   |   |   |   |-- ✅ nodes.py
|   |   |       |   |   |   |   \-- ✅ utils.py
|   |   |       |   |   |   |-- ✅ starknet/
|   |   |       |   |   |   |   |-- ✅ abi/
|   |   |       |   |   |   |   |   |-- ✅ v0/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ model.py
|   |   |       |   |   |   |   |   |   |-- ✅ parser.py
|   |   |       |   |   |   |   |   |   |-- ✅ schemas.py
|   |   |       |   |   |   |   |   |   \-- ✅ shape.py
|   |   |       |   |   |   |   |   |-- ✅ v1/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ core_structures.json
|   |   |       |   |   |   |   |   |   |-- ✅ model.py
|   |   |       |   |   |   |   |   |   |-- ✅ parser.py
|   |   |       |   |   |   |   |   |   |-- ✅ parser_transformer.py
|   |   |       |   |   |   |   |   |   |-- ✅ schemas.py
|   |   |       |   |   |   |   |   |   \-- ✅ shape.py
|   |   |       |   |   |   |   |   \-- ✅ v2/
|   |   |       |   |   |   |   |       |-- ✅ __init__.py
|   |   |       |   |   |   |   |       |-- ✅ model.py
|   |   |       |   |   |   |   |       |-- ✅ parser.py
|   |   |       |   |   |   |   |       |-- ✅ parser_transformer.py
|   |   |       |   |   |   |   |       |-- ✅ schemas.py
|   |   |       |   |   |   |   |       \-- ✅ shape.py
|   |   |       |   |   |   |   |-- ✅ cairo/
|   |   |       |   |   |   |   |   |-- ✅ deprecated_parse/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ cairo_types.py
|   |   |       |   |   |   |   |   |   |-- ✅ parser.py
|   |   |       |   |   |   |   |   |   \-- ✅ parser_transformer.py
|   |   |       |   |   |   |   |   |-- ✅ v1/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   \-- ✅ type_parser.py
|   |   |       |   |   |   |   |   |-- ✅ v2/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   \-- ✅ type_parser.py
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ data_types.py
|   |   |       |   |   |   |   |   |-- ✅ felt.py
|   |   |       |   |   |   |   |   \-- ✅ type_parser.py
|   |   |       |   |   |   |   |-- ✅ hash/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ address.py
|   |   |       |   |   |   |   |   |-- ✅ compiled_class_hash_objects.py
|   |   |       |   |   |   |   |   |-- ✅ selector.py
|   |   |       |   |   |   |   |   |-- ✅ storage.py
|   |   |       |   |   |   |   |   \-- ✅ utils.py
|   |   |       |   |   |   |   |-- ✅ models/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   \-- ✅ typed_data.py
|   |   |       |   |   |   |   |-- ✅ serialization/
|   |   |       |   |   |   |   |   |-- ✅ data_serializers/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ _common.py
|   |   |       |   |   |   |   |   |   |-- ✅ array_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ bool_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ byte_array_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ cairo_data_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ enum_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ felt_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ named_tuple_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ option_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ output_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ payload_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ struct_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ tuple_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ uint256_serializer.py
|   |   |       |   |   |   |   |   |   |-- ✅ uint_serializer.py
|   |   |       |   |   |   |   |   |   \-- ✅ unit_serializer.py
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ _calldata_reader.py
|   |   |       |   |   |   |   |   |-- ✅ _context.py
|   |   |       |   |   |   |   |   |-- ✅ errors.py
|   |   |       |   |   |   |   |   |-- ✅ factory.py
|   |   |       |   |   |   |   |   |-- ✅ function_serialization_adapter.py
|   |   |       |   |   |   |   |   \-- ✅ tuple_dataclass.py
|   |   |       |   |   |   |   |-- ✅ utils/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ constructor_args_translator.py
|   |   |       |   |   |   |   |   |-- ✅ iterable.py
|   |   |       |   |   |   |   |   |-- ✅ schema.py
|   |   |       |   |   |   |   |   \-- ✅ typed_data.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ ccxt_utils.py
|   |   |       |   |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |   \-- ✅ constants.py
|   |   |       |   |   |   |-- ✅ starkware/
|   |   |       |   |   |   |   |-- ✅ crypto/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ fast_pedersen_hash.py
|   |   |       |   |   |   |   |   |-- ✅ math_utils.py
|   |   |       |   |   |   |   |   |-- ✅ signature.py
|   |   |       |   |   |   |   |   \-- ✅ utils.py
|   |   |       |   |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ sympy/
|   |   |       |   |   |   |   |-- ✅ core/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   \-- ✅ intfunc.py
|   |   |       |   |   |   |   |-- ✅ external/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ gmpy.py
|   |   |       |   |   |   |   |   |-- ✅ importtools.py
|   |   |       |   |   |   |   |   |-- ✅ ntheory.py
|   |   |       |   |   |   |   |   \-- ✅ pythonmpq.py
|   |   |       |   |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ toolz/
|   |   |       |   |   |   |   |-- ✅ curried/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |   |   \-- ✅ operator.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _signatures.py
|   |   |       |   |   |   |   |-- ✅ _version.py
|   |   |       |   |   |   |   |-- ✅ compatibility.py
|   |   |       |   |   |   |   |-- ✅ dicttoolz.py
|   |   |       |   |   |   |   |-- ✅ functoolz.py
|   |   |       |   |   |   |   |-- ✅ itertoolz.py
|   |   |       |   |   |   |   |-- ✅ recipes.py
|   |   |       |   |   |   |   \-- ✅ utils.py
|   |   |       |   |   |   |-- ✅ typing_inspect/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ typing_inspect.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ README.md
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ binance.py
|   |   |       |   |-- ✅ ws/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ constants.py
|   |   |       |   |   |-- ✅ depthcache.py
|   |   |       |   |   |-- ✅ keepalive_websocket.py
|   |   |       |   |   |-- ✅ reconnecting_websocket.py
|   |   |       |   |   |-- ✅ streams.py
|   |   |       |   |   |-- ✅ threaded_stream.py
|   |   |       |   |   \-- ✅ websocket_api.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ async_client.py
|   |   |       |   |-- ✅ base_client.py
|   |   |       |   |-- ✅ client.py
|   |   |       |   |-- ✅ enums.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   \-- ✅ helpers.py
|   |   |       |-- ✅ binance-0.3.71.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ certifi/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __main__.py
|   |   |       |   |-- ✅ cacert.pem
|   |   |       |   |-- ✅ core.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ certifi-2025.8.3.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ cffi/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _cffi_errors.h
|   |   |       |   |-- ✅ _cffi_include.h
|   |   |       |   |-- ✅ _embedding.h
|   |   |       |   |-- ✅ _imp_emulation.py
|   |   |       |   |-- ✅ _shimmed_dist_utils.py
|   |   |       |   |-- ✅ api.py
|   |   |       |   |-- ✅ backend_ctypes.py
|   |   |       |   |-- ✅ cffi_opcode.py
|   |   |       |   |-- ✅ commontypes.py
|   |   |       |   |-- ✅ cparser.py
|   |   |       |   |-- ✅ error.py
|   |   |       |   |-- ✅ ffiplatform.py
|   |   |       |   |-- ✅ lock.py
|   |   |       |   |-- ✅ model.py
|   |   |       |   |-- ✅ parse_c_type.h
|   |   |       |   |-- ✅ pkgconfig.py
|   |   |       |   |-- ✅ recompiler.py
|   |   |       |   |-- ✅ setuptools_ext.py
|   |   |       |   |-- ✅ vengine_cpy.py
|   |   |       |   |-- ✅ vengine_gen.py
|   |   |       |   \-- ✅ verifier.py
|   |   |       |-- ✅ cffi-2.0.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ AUTHORS
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ charset_normalizer/
|   |   |       |   |-- ✅ cli/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ __main__.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __main__.py
|   |   |       |   |-- ✅ api.py
|   |   |       |   |-- ✅ cd.py
|   |   |       |   |-- ✅ constant.py
|   |   |       |   |-- ✅ legacy.py
|   |   |       |   |-- ✅ md.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ md.py
|   |   |       |   |-- ✅ md__mypyc.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ models.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ utils.py
|   |   |       |   \-- ✅ version.py
|   |   |       |-- ✅ charset_normalizer-3.4.3.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ click/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _compat.py
|   |   |       |   |-- ✅ _termui_impl.py
|   |   |       |   |-- ✅ _textwrap.py
|   |   |       |   |-- ✅ _utils.py
|   |   |       |   |-- ✅ _winconsole.py
|   |   |       |   |-- ✅ core.py
|   |   |       |   |-- ✅ decorators.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ formatting.py
|   |   |       |   |-- ✅ globals.py
|   |   |       |   |-- ✅ parser.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ shell_completion.py
|   |   |       |   |-- ✅ termui.py
|   |   |       |   |-- ✅ testing.py
|   |   |       |   |-- ✅ types.py
|   |   |       |   \-- ✅ utils.py
|   |   |       |-- ✅ click-8.3.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ colorama/
|   |   |       |   |-- ✅ tests/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ ansi_test.py
|   |   |       |   |   |-- ✅ ansitowin32_test.py
|   |   |       |   |   |-- ✅ initialise_test.py
|   |   |       |   |   |-- ✅ isatty_test.py
|   |   |       |   |   |-- ✅ utils.py
|   |   |       |   |   \-- ✅ winterm_test.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ ansi.py
|   |   |       |   |-- ✅ ansitowin32.py
|   |   |       |   |-- ✅ initialise.py
|   |   |       |   |-- ✅ win32.py
|   |   |       |   \-- ✅ winterm.py
|   |   |       |-- ✅ colorama-0.4.6.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ contourpy/
|   |   |       |   |-- ✅ util/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _build_config.py
|   |   |       |   |   |-- ✅ bokeh_renderer.py
|   |   |       |   |   |-- ✅ bokeh_util.py
|   |   |       |   |   |-- ✅ data.py
|   |   |       |   |   |-- ✅ mpl_renderer.py
|   |   |       |   |   |-- ✅ mpl_util.py
|   |   |       |   |   \-- ✅ renderer.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _contourpy.cp312-win_amd64.lib
|   |   |       |   |-- ✅ _contourpy.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _contourpy.pyi
|   |   |       |   |-- ✅ _version.py
|   |   |       |   |-- ✅ array.py
|   |   |       |   |-- ✅ chunk.py
|   |   |       |   |-- ✅ convert.py
|   |   |       |   |-- ✅ dechunk.py
|   |   |       |   |-- ✅ enum_util.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ typecheck.py
|   |   |       |   \-- ✅ types.py
|   |   |       |-- ✅ contourpy-1.3.3.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ Crypto/
|   |   |       |   |-- ✅ Cipher/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _ARC4.pyd
|   |   |       |   |   |-- ✅ _chacha20.pyd
|   |   |       |   |   |-- ✅ _EKSBlowfish.py
|   |   |       |   |   |-- ✅ _EKSBlowfish.pyi
|   |   |       |   |   |-- ✅ _mode_cbc.py
|   |   |       |   |   |-- ✅ _mode_cbc.pyi
|   |   |       |   |   |-- ✅ _mode_ccm.py
|   |   |       |   |   |-- ✅ _mode_ccm.pyi
|   |   |       |   |   |-- ✅ _mode_cfb.py
|   |   |       |   |   |-- ✅ _mode_cfb.pyi
|   |   |       |   |   |-- ✅ _mode_ctr.py
|   |   |       |   |   |-- ✅ _mode_ctr.pyi
|   |   |       |   |   |-- ✅ _mode_eax.py
|   |   |       |   |   |-- ✅ _mode_eax.pyi
|   |   |       |   |   |-- ✅ _mode_ecb.py
|   |   |       |   |   |-- ✅ _mode_ecb.pyi
|   |   |       |   |   |-- ✅ _mode_gcm.py
|   |   |       |   |   |-- ✅ _mode_gcm.pyi
|   |   |       |   |   |-- ✅ _mode_kw.py
|   |   |       |   |   |-- ✅ _mode_kwp.py
|   |   |       |   |   |-- ✅ _mode_ocb.py
|   |   |       |   |   |-- ✅ _mode_ocb.pyi
|   |   |       |   |   |-- ✅ _mode_ofb.py
|   |   |       |   |   |-- ✅ _mode_ofb.pyi
|   |   |       |   |   |-- ✅ _mode_openpgp.py
|   |   |       |   |   |-- ✅ _mode_openpgp.pyi
|   |   |       |   |   |-- ✅ _mode_siv.py
|   |   |       |   |   |-- ✅ _mode_siv.pyi
|   |   |       |   |   |-- ✅ _pkcs1_decode.pyd
|   |   |       |   |   |-- ✅ _pkcs1_oaep_decode.py
|   |   |       |   |   |-- ✅ _raw_aes.pyd
|   |   |       |   |   |-- ✅ _raw_aesni.pyd
|   |   |       |   |   |-- ✅ _raw_arc2.pyd
|   |   |       |   |   |-- ✅ _raw_blowfish.pyd
|   |   |       |   |   |-- ✅ _raw_cast.pyd
|   |   |       |   |   |-- ✅ _raw_cbc.pyd
|   |   |       |   |   |-- ✅ _raw_cfb.pyd
|   |   |       |   |   |-- ✅ _raw_ctr.pyd
|   |   |       |   |   |-- ✅ _raw_des.pyd
|   |   |       |   |   |-- ✅ _raw_des3.pyd
|   |   |       |   |   |-- ✅ _raw_ecb.pyd
|   |   |       |   |   |-- ✅ _raw_eksblowfish.pyd
|   |   |       |   |   |-- ✅ _raw_ocb.pyd
|   |   |       |   |   |-- ✅ _raw_ofb.pyd
|   |   |       |   |   |-- ✅ _Salsa20.pyd
|   |   |       |   |   |-- ✅ AES.py
|   |   |       |   |   |-- ✅ AES.pyi
|   |   |       |   |   |-- ✅ ARC2.py
|   |   |       |   |   |-- ✅ ARC2.pyi
|   |   |       |   |   |-- ✅ ARC4.py
|   |   |       |   |   |-- ✅ ARC4.pyi
|   |   |       |   |   |-- ✅ Blowfish.py
|   |   |       |   |   |-- ✅ Blowfish.pyi
|   |   |       |   |   |-- ✅ CAST.py
|   |   |       |   |   |-- ✅ CAST.pyi
|   |   |       |   |   |-- ✅ ChaCha20.py
|   |   |       |   |   |-- ✅ ChaCha20.pyi
|   |   |       |   |   |-- ✅ ChaCha20_Poly1305.py
|   |   |       |   |   |-- ✅ ChaCha20_Poly1305.pyi
|   |   |       |   |   |-- ✅ DES.py
|   |   |       |   |   |-- ✅ DES.pyi
|   |   |       |   |   |-- ✅ DES3.py
|   |   |       |   |   |-- ✅ DES3.pyi
|   |   |       |   |   |-- ✅ PKCS1_OAEP.py
|   |   |       |   |   |-- ✅ PKCS1_OAEP.pyi
|   |   |       |   |   |-- ✅ PKCS1_v1_5.py
|   |   |       |   |   |-- ✅ PKCS1_v1_5.pyi
|   |   |       |   |   |-- ✅ Salsa20.py
|   |   |       |   |   \-- ✅ Salsa20.pyi
|   |   |       |   |-- ✅ Hash/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _BLAKE2b.pyd
|   |   |       |   |   |-- ✅ _BLAKE2s.pyd
|   |   |       |   |   |-- ✅ _ghash_clmul.pyd
|   |   |       |   |   |-- ✅ _ghash_portable.pyd
|   |   |       |   |   |-- ✅ _keccak.pyd
|   |   |       |   |   |-- ✅ _MD2.pyd
|   |   |       |   |   |-- ✅ _MD4.pyd
|   |   |       |   |   |-- ✅ _MD5.pyd
|   |   |       |   |   |-- ✅ _poly1305.pyd
|   |   |       |   |   |-- ✅ _RIPEMD160.pyd
|   |   |       |   |   |-- ✅ _SHA1.pyd
|   |   |       |   |   |-- ✅ _SHA224.pyd
|   |   |       |   |   |-- ✅ _SHA256.pyd
|   |   |       |   |   |-- ✅ _SHA384.pyd
|   |   |       |   |   |-- ✅ _SHA512.pyd
|   |   |       |   |   |-- ✅ BLAKE2b.py
|   |   |       |   |   |-- ✅ BLAKE2b.pyi
|   |   |       |   |   |-- ✅ BLAKE2s.py
|   |   |       |   |   |-- ✅ BLAKE2s.pyi
|   |   |       |   |   |-- ✅ CMAC.py
|   |   |       |   |   |-- ✅ CMAC.pyi
|   |   |       |   |   |-- ✅ cSHAKE128.py
|   |   |       |   |   |-- ✅ cSHAKE128.pyi
|   |   |       |   |   |-- ✅ cSHAKE256.py
|   |   |       |   |   |-- ✅ cSHAKE256.pyi
|   |   |       |   |   |-- ✅ HMAC.py
|   |   |       |   |   |-- ✅ HMAC.pyi
|   |   |       |   |   |-- ✅ KangarooTwelve.py
|   |   |       |   |   |-- ✅ KangarooTwelve.pyi
|   |   |       |   |   |-- ✅ keccak.py
|   |   |       |   |   |-- ✅ keccak.pyi
|   |   |       |   |   |-- ✅ KMAC128.py
|   |   |       |   |   |-- ✅ KMAC128.pyi
|   |   |       |   |   |-- ✅ KMAC256.py
|   |   |       |   |   |-- ✅ KMAC256.pyi
|   |   |       |   |   |-- ✅ MD2.py
|   |   |       |   |   |-- ✅ MD2.pyi
|   |   |       |   |   |-- ✅ MD4.py
|   |   |       |   |   |-- ✅ MD4.pyi
|   |   |       |   |   |-- ✅ MD5.py
|   |   |       |   |   |-- ✅ MD5.pyi
|   |   |       |   |   |-- ✅ Poly1305.py
|   |   |       |   |   |-- ✅ Poly1305.pyi
|   |   |       |   |   |-- ✅ RIPEMD.py
|   |   |       |   |   |-- ✅ RIPEMD.pyi
|   |   |       |   |   |-- ✅ RIPEMD160.py
|   |   |       |   |   |-- ✅ RIPEMD160.pyi
|   |   |       |   |   |-- ✅ SHA.py
|   |   |       |   |   |-- ✅ SHA.pyi
|   |   |       |   |   |-- ✅ SHA1.py
|   |   |       |   |   |-- ✅ SHA1.pyi
|   |   |       |   |   |-- ✅ SHA224.py
|   |   |       |   |   |-- ✅ SHA224.pyi
|   |   |       |   |   |-- ✅ SHA256.py
|   |   |       |   |   |-- ✅ SHA256.pyi
|   |   |       |   |   |-- ✅ SHA384.py
|   |   |       |   |   |-- ✅ SHA384.pyi
|   |   |       |   |   |-- ✅ SHA3_224.py
|   |   |       |   |   |-- ✅ SHA3_224.pyi
|   |   |       |   |   |-- ✅ SHA3_256.py
|   |   |       |   |   |-- ✅ SHA3_256.pyi
|   |   |       |   |   |-- ✅ SHA3_384.py
|   |   |       |   |   |-- ✅ SHA3_384.pyi
|   |   |       |   |   |-- ✅ SHA3_512.py
|   |   |       |   |   |-- ✅ SHA3_512.pyi
|   |   |       |   |   |-- ✅ SHA512.py
|   |   |       |   |   |-- ✅ SHA512.pyi
|   |   |       |   |   |-- ✅ SHAKE128.py
|   |   |       |   |   |-- ✅ SHAKE128.pyi
|   |   |       |   |   |-- ✅ SHAKE256.py
|   |   |       |   |   |-- ✅ SHAKE256.pyi
|   |   |       |   |   |-- ✅ TupleHash128.py
|   |   |       |   |   |-- ✅ TupleHash128.pyi
|   |   |       |   |   |-- ✅ TupleHash256.py
|   |   |       |   |   |-- ✅ TupleHash256.pyi
|   |   |       |   |   |-- ✅ TurboSHAKE128.py
|   |   |       |   |   |-- ✅ TurboSHAKE128.pyi
|   |   |       |   |   |-- ✅ TurboSHAKE256.py
|   |   |       |   |   \-- ✅ TurboSHAKE256.pyi
|   |   |       |   |-- ✅ IO/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _PBES.py
|   |   |       |   |   |-- ✅ _PBES.pyi
|   |   |       |   |   |-- ✅ PEM.py
|   |   |       |   |   |-- ✅ PEM.pyi
|   |   |       |   |   |-- ✅ PKCS8.py
|   |   |       |   |   \-- ✅ PKCS8.pyi
|   |   |       |   |-- ✅ Math/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _IntegerBase.py
|   |   |       |   |   |-- ✅ _IntegerBase.pyi
|   |   |       |   |   |-- ✅ _IntegerCustom.py
|   |   |       |   |   |-- ✅ _IntegerCustom.pyi
|   |   |       |   |   |-- ✅ _IntegerGMP.py
|   |   |       |   |   |-- ✅ _IntegerGMP.pyi
|   |   |       |   |   |-- ✅ _IntegerNative.py
|   |   |       |   |   |-- ✅ _IntegerNative.pyi
|   |   |       |   |   |-- ✅ _modexp.pyd
|   |   |       |   |   |-- ✅ Numbers.py
|   |   |       |   |   |-- ✅ Numbers.pyi
|   |   |       |   |   |-- ✅ Primality.py
|   |   |       |   |   \-- ✅ Primality.pyi
|   |   |       |   |-- ✅ Protocol/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _scrypt.pyd
|   |   |       |   |   |-- ✅ DH.py
|   |   |       |   |   |-- ✅ DH.pyi
|   |   |       |   |   |-- ✅ HPKE.py
|   |   |       |   |   |-- ✅ KDF.py
|   |   |       |   |   |-- ✅ KDF.pyi
|   |   |       |   |   |-- ✅ SecretSharing.py
|   |   |       |   |   \-- ✅ SecretSharing.pyi
|   |   |       |   |-- ✅ PublicKey/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _curve.py
|   |   |       |   |   |-- ✅ _curve25519.pyd
|   |   |       |   |   |-- ✅ _curve448.pyd
|   |   |       |   |   |-- ✅ _ec_ws.pyd
|   |   |       |   |   |-- ✅ _ed25519.pyd
|   |   |       |   |   |-- ✅ _ed448.pyd
|   |   |       |   |   |-- ✅ _edwards.py
|   |   |       |   |   |-- ✅ _montgomery.py
|   |   |       |   |   |-- ✅ _nist_ecc.py
|   |   |       |   |   |-- ✅ _openssh.py
|   |   |       |   |   |-- ✅ _openssh.pyi
|   |   |       |   |   |-- ✅ _point.py
|   |   |       |   |   |-- ✅ _point.pyi
|   |   |       |   |   |-- ✅ DSA.py
|   |   |       |   |   |-- ✅ DSA.pyi
|   |   |       |   |   |-- ✅ ECC.py
|   |   |       |   |   |-- ✅ ECC.pyi
|   |   |       |   |   |-- ✅ ElGamal.py
|   |   |       |   |   |-- ✅ ElGamal.pyi
|   |   |       |   |   |-- ✅ RSA.py
|   |   |       |   |   \-- ✅ RSA.pyi
|   |   |       |   |-- ✅ Random/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ random.py
|   |   |       |   |   \-- ✅ random.pyi
|   |   |       |   |-- ✅ SelfTest/
|   |   |       |   |   |-- ✅ Cipher/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ test_AES.py
|   |   |       |   |   |   |-- ✅ test_ARC2.py
|   |   |       |   |   |   |-- ✅ test_ARC4.py
|   |   |       |   |   |   |-- ✅ test_Blowfish.py
|   |   |       |   |   |   |-- ✅ test_CAST.py
|   |   |       |   |   |   |-- ✅ test_CBC.py
|   |   |       |   |   |   |-- ✅ test_CCM.py
|   |   |       |   |   |   |-- ✅ test_CFB.py
|   |   |       |   |   |   |-- ✅ test_ChaCha20.py
|   |   |       |   |   |   |-- ✅ test_ChaCha20_Poly1305.py
|   |   |       |   |   |   |-- ✅ test_CTR.py
|   |   |       |   |   |   |-- ✅ test_DES.py
|   |   |       |   |   |   |-- ✅ test_DES3.py
|   |   |       |   |   |   |-- ✅ test_EAX.py
|   |   |       |   |   |   |-- ✅ test_GCM.py
|   |   |       |   |   |   |-- ✅ test_KW.py
|   |   |       |   |   |   |-- ✅ test_OCB.py
|   |   |       |   |   |   |-- ✅ test_OFB.py
|   |   |       |   |   |   |-- ✅ test_OpenPGP.py
|   |   |       |   |   |   |-- ✅ test_pkcs1_15.py
|   |   |       |   |   |   |-- ✅ test_pkcs1_oaep.py
|   |   |       |   |   |   |-- ✅ test_Salsa20.py
|   |   |       |   |   |   \-- ✅ test_SIV.py
|   |   |       |   |   |-- ✅ Hash/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ test_BLAKE2.py
|   |   |       |   |   |   |-- ✅ test_CMAC.py
|   |   |       |   |   |   |-- ✅ test_cSHAKE.py
|   |   |       |   |   |   |-- ✅ test_HMAC.py
|   |   |       |   |   |   |-- ✅ test_KangarooTwelve.py
|   |   |       |   |   |   |-- ✅ test_keccak.py
|   |   |       |   |   |   |-- ✅ test_KMAC.py
|   |   |       |   |   |   |-- ✅ test_MD2.py
|   |   |       |   |   |   |-- ✅ test_MD4.py
|   |   |       |   |   |   |-- ✅ test_MD5.py
|   |   |       |   |   |   |-- ✅ test_Poly1305.py
|   |   |       |   |   |   |-- ✅ test_RIPEMD160.py
|   |   |       |   |   |   |-- ✅ test_SHA1.py
|   |   |       |   |   |   |-- ✅ test_SHA224.py
|   |   |       |   |   |   |-- ✅ test_SHA256.py
|   |   |       |   |   |   |-- ✅ test_SHA384.py
|   |   |       |   |   |   |-- ✅ test_SHA3_224.py
|   |   |       |   |   |   |-- ✅ test_SHA3_256.py
|   |   |       |   |   |   |-- ✅ test_SHA3_384.py
|   |   |       |   |   |   |-- ✅ test_SHA3_512.py
|   |   |       |   |   |   |-- ✅ test_SHA512.py
|   |   |       |   |   |   |-- ✅ test_SHAKE.py
|   |   |       |   |   |   |-- ✅ test_TupleHash.py
|   |   |       |   |   |   \-- ✅ test_TurboSHAKE.py
|   |   |       |   |   |-- ✅ IO/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_PBES.py
|   |   |       |   |   |   \-- ✅ test_PKCS8.py
|   |   |       |   |   |-- ✅ Math/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_modexp.py
|   |   |       |   |   |   |-- ✅ test_modmult.py
|   |   |       |   |   |   |-- ✅ test_Numbers.py
|   |   |       |   |   |   \-- ✅ test_Primality.py
|   |   |       |   |   |-- ✅ Protocol/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_ecdh.py
|   |   |       |   |   |   |-- ✅ test_HPKE.py
|   |   |       |   |   |   |-- ✅ test_KDF.py
|   |   |       |   |   |   |-- ✅ test_rfc1751.py
|   |   |       |   |   |   \-- ✅ test_SecretSharing.py
|   |   |       |   |   |-- ✅ PublicKey/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_DSA.py
|   |   |       |   |   |   |-- ✅ test_ECC_Curve25519.py
|   |   |       |   |   |   |-- ✅ test_ECC_Curve448.py
|   |   |       |   |   |   |-- ✅ test_ECC_Ed25519.py
|   |   |       |   |   |   |-- ✅ test_ECC_Ed448.py
|   |   |       |   |   |   |-- ✅ test_ECC_NIST.py
|   |   |       |   |   |   |-- ✅ test_ElGamal.py
|   |   |       |   |   |   |-- ✅ test_import_Curve25519.py
|   |   |       |   |   |   |-- ✅ test_import_Curve448.py
|   |   |       |   |   |   |-- ✅ test_import_DSA.py
|   |   |       |   |   |   |-- ✅ test_import_ECC.py
|   |   |       |   |   |   |-- ✅ test_import_RSA.py
|   |   |       |   |   |   \-- ✅ test_RSA.py
|   |   |       |   |   |-- ✅ Random/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ test_random.py
|   |   |       |   |   |-- ✅ Signature/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_dss.py
|   |   |       |   |   |   |-- ✅ test_eddsa.py
|   |   |       |   |   |   |-- ✅ test_pkcs1_15.py
|   |   |       |   |   |   \-- ✅ test_pss.py
|   |   |       |   |   |-- ✅ Util/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_asn1.py
|   |   |       |   |   |   |-- ✅ test_Counter.py
|   |   |       |   |   |   |-- ✅ test_number.py
|   |   |       |   |   |   |-- ✅ test_Padding.py
|   |   |       |   |   |   |-- ✅ test_rfc1751.py
|   |   |       |   |   |   \-- ✅ test_strxor.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __main__.py
|   |   |       |   |   |-- ✅ loader.py
|   |   |       |   |   \-- ✅ st_common.py
|   |   |       |   |-- ✅ Signature/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ DSS.py
|   |   |       |   |   |-- ✅ DSS.pyi
|   |   |       |   |   |-- ✅ eddsa.py
|   |   |       |   |   |-- ✅ eddsa.pyi
|   |   |       |   |   |-- ✅ pkcs1_15.py
|   |   |       |   |   |-- ✅ pkcs1_15.pyi
|   |   |       |   |   |-- ✅ PKCS1_PSS.py
|   |   |       |   |   |-- ✅ PKCS1_PSS.pyi
|   |   |       |   |   |-- ✅ PKCS1_v1_5.py
|   |   |       |   |   |-- ✅ PKCS1_v1_5.pyi
|   |   |       |   |   |-- ✅ pss.py
|   |   |       |   |   \-- ✅ pss.pyi
|   |   |       |   |-- ✅ Util/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _cpu_features.py
|   |   |       |   |   |-- ✅ _cpu_features.pyi
|   |   |       |   |   |-- ✅ _cpuid_c.pyd
|   |   |       |   |   |-- ✅ _file_system.py
|   |   |       |   |   |-- ✅ _file_system.pyi
|   |   |       |   |   |-- ✅ _raw_api.py
|   |   |       |   |   |-- ✅ _raw_api.pyi
|   |   |       |   |   |-- ✅ _strxor.pyd
|   |   |       |   |   |-- ✅ asn1.py
|   |   |       |   |   |-- ✅ asn1.pyi
|   |   |       |   |   |-- ✅ Counter.py
|   |   |       |   |   |-- ✅ Counter.pyi
|   |   |       |   |   |-- ✅ number.py
|   |   |       |   |   |-- ✅ number.pyi
|   |   |       |   |   |-- ✅ Padding.py
|   |   |       |   |   |-- ✅ Padding.pyi
|   |   |       |   |   |-- ✅ py3compat.py
|   |   |       |   |   |-- ✅ py3compat.pyi
|   |   |       |   |   |-- ✅ RFC1751.py
|   |   |       |   |   |-- ✅ RFC1751.pyi
|   |   |       |   |   |-- ✅ strxor.py
|   |   |       |   |   \-- ✅ strxor.pyi
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __init__.pyi
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ cryptography/
|   |   |       |   |-- ✅ hazmat/
|   |   |       |   |   |-- ✅ asn1/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ asn1.py
|   |   |       |   |   |-- ✅ backends/
|   |   |       |   |   |   |-- ✅ openssl/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ backend.py
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ bindings/
|   |   |       |   |   |   |-- ✅ _rust/
|   |   |       |   |   |   |   |-- ✅ openssl/
|   |   |       |   |   |   |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |   |   |   |-- ✅ aead.pyi
|   |   |       |   |   |   |   |   |-- ✅ ciphers.pyi
|   |   |       |   |   |   |   |   |-- ✅ cmac.pyi
|   |   |       |   |   |   |   |   |-- ✅ dh.pyi
|   |   |       |   |   |   |   |   |-- ✅ dsa.pyi
|   |   |       |   |   |   |   |   |-- ✅ ec.pyi
|   |   |       |   |   |   |   |   |-- ✅ ed25519.pyi
|   |   |       |   |   |   |   |   |-- ✅ ed448.pyi
|   |   |       |   |   |   |   |   |-- ✅ hashes.pyi
|   |   |       |   |   |   |   |   |-- ✅ hmac.pyi
|   |   |       |   |   |   |   |   |-- ✅ kdf.pyi
|   |   |       |   |   |   |   |   |-- ✅ keys.pyi
|   |   |       |   |   |   |   |   |-- ✅ poly1305.pyi
|   |   |       |   |   |   |   |   |-- ✅ rsa.pyi
|   |   |       |   |   |   |   |   |-- ✅ x25519.pyi
|   |   |       |   |   |   |   |   \-- ✅ x448.pyi
|   |   |       |   |   |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |   |   |-- ✅ _openssl.pyi
|   |   |       |   |   |   |   |-- ✅ asn1.pyi
|   |   |       |   |   |   |   |-- ✅ declarative_asn1.pyi
|   |   |       |   |   |   |   |-- ✅ exceptions.pyi
|   |   |       |   |   |   |   |-- ✅ ocsp.pyi
|   |   |       |   |   |   |   |-- ✅ pkcs12.pyi
|   |   |       |   |   |   |   |-- ✅ pkcs7.pyi
|   |   |       |   |   |   |   |-- ✅ test_support.pyi
|   |   |       |   |   |   |   \-- ✅ x509.pyi
|   |   |       |   |   |   |-- ✅ openssl/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _conditional.py
|   |   |       |   |   |   |   \-- ✅ binding.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ _rust.pyd
|   |   |       |   |   |-- ✅ decrepit/
|   |   |       |   |   |   |-- ✅ ciphers/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ algorithms.py
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ primitives/
|   |   |       |   |   |   |-- ✅ asymmetric/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ dh.py
|   |   |       |   |   |   |   |-- ✅ dsa.py
|   |   |       |   |   |   |   |-- ✅ ec.py
|   |   |       |   |   |   |   |-- ✅ ed25519.py
|   |   |       |   |   |   |   |-- ✅ ed448.py
|   |   |       |   |   |   |   |-- ✅ padding.py
|   |   |       |   |   |   |   |-- ✅ rsa.py
|   |   |       |   |   |   |   |-- ✅ types.py
|   |   |       |   |   |   |   |-- ✅ utils.py
|   |   |       |   |   |   |   |-- ✅ x25519.py
|   |   |       |   |   |   |   \-- ✅ x448.py
|   |   |       |   |   |   |-- ✅ ciphers/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ aead.py
|   |   |       |   |   |   |   |-- ✅ algorithms.py
|   |   |       |   |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |   \-- ✅ modes.py
|   |   |       |   |   |   |-- ✅ kdf/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ argon2.py
|   |   |       |   |   |   |   |-- ✅ concatkdf.py
|   |   |       |   |   |   |   |-- ✅ hkdf.py
|   |   |       |   |   |   |   |-- ✅ kbkdf.py
|   |   |       |   |   |   |   |-- ✅ pbkdf2.py
|   |   |       |   |   |   |   |-- ✅ scrypt.py
|   |   |       |   |   |   |   \-- ✅ x963kdf.py
|   |   |       |   |   |   |-- ✅ serialization/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |   |-- ✅ pkcs12.py
|   |   |       |   |   |   |   |-- ✅ pkcs7.py
|   |   |       |   |   |   |   \-- ✅ ssh.py
|   |   |       |   |   |   |-- ✅ twofactor/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ hotp.py
|   |   |       |   |   |   |   \-- ✅ totp.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _asymmetric.py
|   |   |       |   |   |   |-- ✅ _cipheralgorithm.py
|   |   |       |   |   |   |-- ✅ _serialization.py
|   |   |       |   |   |   |-- ✅ cmac.py
|   |   |       |   |   |   |-- ✅ constant_time.py
|   |   |       |   |   |   |-- ✅ hashes.py
|   |   |       |   |   |   |-- ✅ hmac.py
|   |   |       |   |   |   |-- ✅ keywrap.py
|   |   |       |   |   |   |-- ✅ padding.py
|   |   |       |   |   |   \-- ✅ poly1305.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ _oid.py
|   |   |       |   |-- ✅ x509/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ certificate_transparency.py
|   |   |       |   |   |-- ✅ extensions.py
|   |   |       |   |   |-- ✅ general_name.py
|   |   |       |   |   |-- ✅ name.py
|   |   |       |   |   |-- ✅ ocsp.py
|   |   |       |   |   |-- ✅ oid.py
|   |   |       |   |   \-- ✅ verification.py
|   |   |       |   |-- ✅ __about__.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ fernet.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   \-- ✅ utils.py
|   |   |       |-- ✅ cryptography-46.0.1.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ LICENSE
|   |   |       |   |   |-- ✅ LICENSE.APACHE
|   |   |       |   |   \-- ✅ LICENSE.BSD
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ cycler/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ cycler-0.12.1.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ dateparser/
|   |   |       |   |-- ✅ calendars/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ hijri.py
|   |   |       |   |   |-- ✅ hijri_parser.py
|   |   |       |   |   |-- ✅ jalali.py
|   |   |       |   |   \-- ✅ jalali_parser.py
|   |   |       |   |-- ✅ custom_language_detection/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ fasttext.py
|   |   |       |   |   |-- ✅ langdetect.py
|   |   |       |   |   \-- ✅ language_mapping.py
|   |   |       |   |-- ✅ data/
|   |   |       |   |   |-- ✅ date_translation_data/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ af.py
|   |   |       |   |   |   |-- ✅ agq.py
|   |   |       |   |   |   |-- ✅ ak.py
|   |   |       |   |   |   |-- ✅ am.py
|   |   |       |   |   |   |-- ✅ ar.py
|   |   |       |   |   |   |-- ✅ as.py
|   |   |       |   |   |   |-- ✅ asa.py
|   |   |       |   |   |   |-- ✅ ast.py
|   |   |       |   |   |   |-- ✅ az-Cyrl.py
|   |   |       |   |   |   |-- ✅ az-Latn.py
|   |   |       |   |   |   |-- ✅ az.py
|   |   |       |   |   |   |-- ✅ bas.py
|   |   |       |   |   |   |-- ✅ be.py
|   |   |       |   |   |   |-- ✅ bem.py
|   |   |       |   |   |   |-- ✅ bez.py
|   |   |       |   |   |   |-- ✅ bg.py
|   |   |       |   |   |   |-- ✅ bm.py
|   |   |       |   |   |   |-- ✅ bn.py
|   |   |       |   |   |   |-- ✅ bo.py
|   |   |       |   |   |   |-- ✅ br.py
|   |   |       |   |   |   |-- ✅ brx.py
|   |   |       |   |   |   |-- ✅ bs-Cyrl.py
|   |   |       |   |   |   |-- ✅ bs-Latn.py
|   |   |       |   |   |   |-- ✅ bs.py
|   |   |       |   |   |   |-- ✅ ca.py
|   |   |       |   |   |   |-- ✅ ce.py
|   |   |       |   |   |   |-- ✅ cgg.py
|   |   |       |   |   |   |-- ✅ chr.py
|   |   |       |   |   |   |-- ✅ ckb.py
|   |   |       |   |   |   |-- ✅ cs.py
|   |   |       |   |   |   |-- ✅ cy.py
|   |   |       |   |   |   |-- ✅ da.py
|   |   |       |   |   |   |-- ✅ dav.py
|   |   |       |   |   |   |-- ✅ de.py
|   |   |       |   |   |   |-- ✅ dje.py
|   |   |       |   |   |   |-- ✅ dsb.py
|   |   |       |   |   |   |-- ✅ dua.py
|   |   |       |   |   |   |-- ✅ dyo.py
|   |   |       |   |   |   |-- ✅ dz.py
|   |   |       |   |   |   |-- ✅ ebu.py
|   |   |       |   |   |   |-- ✅ ee.py
|   |   |       |   |   |   |-- ✅ el.py
|   |   |       |   |   |   |-- ✅ en.py
|   |   |       |   |   |   |-- ✅ eo.py
|   |   |       |   |   |   |-- ✅ es.py
|   |   |       |   |   |   |-- ✅ et.py
|   |   |       |   |   |   |-- ✅ eu.py
|   |   |       |   |   |   |-- ✅ ewo.py
|   |   |       |   |   |   |-- ✅ fa.py
|   |   |       |   |   |   |-- ✅ ff.py
|   |   |       |   |   |   |-- ✅ fi.py
|   |   |       |   |   |   |-- ✅ fil.py
|   |   |       |   |   |   |-- ✅ fo.py
|   |   |       |   |   |   |-- ✅ fr.py
|   |   |       |   |   |   |-- ✅ fur.py
|   |   |       |   |   |   |-- ✅ fy.py
|   |   |       |   |   |   |-- ✅ ga.py
|   |   |       |   |   |   |-- ✅ gd.py
|   |   |       |   |   |   |-- ✅ gl.py
|   |   |       |   |   |   |-- ✅ gsw.py
|   |   |       |   |   |   |-- ✅ gu.py
|   |   |       |   |   |   |-- ✅ guz.py
|   |   |       |   |   |   |-- ✅ gv.py
|   |   |       |   |   |   |-- ✅ ha.py
|   |   |       |   |   |   |-- ✅ haw.py
|   |   |       |   |   |   |-- ✅ he.py
|   |   |       |   |   |   |-- ✅ hi.py
|   |   |       |   |   |   |-- ✅ hr.py
|   |   |       |   |   |   |-- ✅ hsb.py
|   |   |       |   |   |   |-- ✅ hu.py
|   |   |       |   |   |   |-- ✅ hy.py
|   |   |       |   |   |   |-- ✅ id.py
|   |   |       |   |   |   |-- ✅ ig.py
|   |   |       |   |   |   |-- ✅ ii.py
|   |   |       |   |   |   |-- ✅ is.py
|   |   |       |   |   |   |-- ✅ it.py
|   |   |       |   |   |   |-- ✅ ja.py
|   |   |       |   |   |   |-- ✅ jgo.py
|   |   |       |   |   |   |-- ✅ jmc.py
|   |   |       |   |   |   |-- ✅ ka.py
|   |   |       |   |   |   |-- ✅ kab.py
|   |   |       |   |   |   |-- ✅ kam.py
|   |   |       |   |   |   |-- ✅ kde.py
|   |   |       |   |   |   |-- ✅ kea.py
|   |   |       |   |   |   |-- ✅ khq.py
|   |   |       |   |   |   |-- ✅ ki.py
|   |   |       |   |   |   |-- ✅ kk.py
|   |   |       |   |   |   |-- ✅ kl.py
|   |   |       |   |   |   |-- ✅ kln.py
|   |   |       |   |   |   |-- ✅ km.py
|   |   |       |   |   |   |-- ✅ kn.py
|   |   |       |   |   |   |-- ✅ ko.py
|   |   |       |   |   |   |-- ✅ kok.py
|   |   |       |   |   |   |-- ✅ ks.py
|   |   |       |   |   |   |-- ✅ ksb.py
|   |   |       |   |   |   |-- ✅ ksf.py
|   |   |       |   |   |   |-- ✅ ksh.py
|   |   |       |   |   |   |-- ✅ kw.py
|   |   |       |   |   |   |-- ✅ ky.py
|   |   |       |   |   |   |-- ✅ lag.py
|   |   |       |   |   |   |-- ✅ lb.py
|   |   |       |   |   |   |-- ✅ lg.py
|   |   |       |   |   |   |-- ✅ lkt.py
|   |   |       |   |   |   |-- ✅ ln.py
|   |   |       |   |   |   |-- ✅ lo.py
|   |   |       |   |   |   |-- ✅ lrc.py
|   |   |       |   |   |   |-- ✅ lt.py
|   |   |       |   |   |   |-- ✅ lu.py
|   |   |       |   |   |   |-- ✅ luo.py
|   |   |       |   |   |   |-- ✅ luy.py
|   |   |       |   |   |   |-- ✅ lv.py
|   |   |       |   |   |   |-- ✅ mas.py
|   |   |       |   |   |   |-- ✅ mer.py
|   |   |       |   |   |   |-- ✅ mfe.py
|   |   |       |   |   |   |-- ✅ mg.py
|   |   |       |   |   |   |-- ✅ mgh.py
|   |   |       |   |   |   |-- ✅ mgo.py
|   |   |       |   |   |   |-- ✅ mk.py
|   |   |       |   |   |   |-- ✅ ml.py
|   |   |       |   |   |   |-- ✅ mn.py
|   |   |       |   |   |   |-- ✅ mr.py
|   |   |       |   |   |   |-- ✅ ms.py
|   |   |       |   |   |   |-- ✅ mt.py
|   |   |       |   |   |   |-- ✅ mua.py
|   |   |       |   |   |   |-- ✅ my.py
|   |   |       |   |   |   |-- ✅ mzn.py
|   |   |       |   |   |   |-- ✅ naq.py
|   |   |       |   |   |   |-- ✅ nb.py
|   |   |       |   |   |   |-- ✅ nd.py
|   |   |       |   |   |   |-- ✅ ne.py
|   |   |       |   |   |   |-- ✅ nl.py
|   |   |       |   |   |   |-- ✅ nmg.py
|   |   |       |   |   |   |-- ✅ nn.py
|   |   |       |   |   |   |-- ✅ nnh.py
|   |   |       |   |   |   |-- ✅ nus.py
|   |   |       |   |   |   |-- ✅ nyn.py
|   |   |       |   |   |   |-- ✅ om.py
|   |   |       |   |   |   |-- ✅ or.py
|   |   |       |   |   |   |-- ✅ os.py
|   |   |       |   |   |   |-- ✅ pa-Arab.py
|   |   |       |   |   |   |-- ✅ pa-Guru.py
|   |   |       |   |   |   |-- ✅ pa.py
|   |   |       |   |   |   |-- ✅ pl.py
|   |   |       |   |   |   |-- ✅ ps.py
|   |   |       |   |   |   |-- ✅ pt.py
|   |   |       |   |   |   |-- ✅ qu.py
|   |   |       |   |   |   |-- ✅ rm.py
|   |   |       |   |   |   |-- ✅ rn.py
|   |   |       |   |   |   |-- ✅ ro.py
|   |   |       |   |   |   |-- ✅ rof.py
|   |   |       |   |   |   |-- ✅ ru.py
|   |   |       |   |   |   |-- ✅ rw.py
|   |   |       |   |   |   |-- ✅ rwk.py
|   |   |       |   |   |   |-- ✅ sah.py
|   |   |       |   |   |   |-- ✅ saq.py
|   |   |       |   |   |   |-- ✅ sbp.py
|   |   |       |   |   |   |-- ✅ se.py
|   |   |       |   |   |   |-- ✅ seh.py
|   |   |       |   |   |   |-- ✅ ses.py
|   |   |       |   |   |   |-- ✅ sg.py
|   |   |       |   |   |   |-- ✅ shi-Latn.py
|   |   |       |   |   |   |-- ✅ shi-Tfng.py
|   |   |       |   |   |   |-- ✅ shi.py
|   |   |       |   |   |   |-- ✅ si.py
|   |   |       |   |   |   |-- ✅ sk.py
|   |   |       |   |   |   |-- ✅ sl.py
|   |   |       |   |   |   |-- ✅ smn.py
|   |   |       |   |   |   |-- ✅ sn.py
|   |   |       |   |   |   |-- ✅ so.py
|   |   |       |   |   |   |-- ✅ sq.py
|   |   |       |   |   |   |-- ✅ sr-Cyrl.py
|   |   |       |   |   |   |-- ✅ sr-Latn.py
|   |   |       |   |   |   |-- ✅ sr.py
|   |   |       |   |   |   |-- ✅ sv.py
|   |   |       |   |   |   |-- ✅ sw.py
|   |   |       |   |   |   |-- ✅ ta.py
|   |   |       |   |   |   |-- ✅ te.py
|   |   |       |   |   |   |-- ✅ teo.py
|   |   |       |   |   |   |-- ✅ th.py
|   |   |       |   |   |   |-- ✅ ti.py
|   |   |       |   |   |   |-- ✅ tl.py
|   |   |       |   |   |   |-- ✅ to.py
|   |   |       |   |   |   |-- ✅ tr.py
|   |   |       |   |   |   |-- ✅ twq.py
|   |   |       |   |   |   |-- ✅ tzm.py
|   |   |       |   |   |   |-- ✅ ug.py
|   |   |       |   |   |   |-- ✅ uk.py
|   |   |       |   |   |   |-- ✅ ur.py
|   |   |       |   |   |   |-- ✅ uz-Arab.py
|   |   |       |   |   |   |-- ✅ uz-Cyrl.py
|   |   |       |   |   |   |-- ✅ uz-Latn.py
|   |   |       |   |   |   |-- ✅ uz.py
|   |   |       |   |   |   |-- ✅ vi.py
|   |   |       |   |   |   |-- ✅ vun.py
|   |   |       |   |   |   |-- ✅ wae.py
|   |   |       |   |   |   |-- ✅ xog.py
|   |   |       |   |   |   |-- ✅ yav.py
|   |   |       |   |   |   |-- ✅ yi.py
|   |   |       |   |   |   |-- ✅ yo.py
|   |   |       |   |   |   |-- ✅ yue.py
|   |   |       |   |   |   |-- ✅ zgh.py
|   |   |       |   |   |   |-- ✅ zh-Hans.py
|   |   |       |   |   |   |-- ✅ zh-Hant.py
|   |   |       |   |   |   |-- ✅ zh.py
|   |   |       |   |   |   \-- ✅ zu.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ dateparser_tz_cache.pkl
|   |   |       |   |   \-- ✅ languages_info.py
|   |   |       |   |-- ✅ languages/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ dictionary.py
|   |   |       |   |   |-- ✅ loader.py
|   |   |       |   |   |-- ✅ locale.py
|   |   |       |   |   \-- ✅ validation.py
|   |   |       |   |-- ✅ search/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ detection.py
|   |   |       |   |   |-- ✅ search.py
|   |   |       |   |   \-- ✅ text_detection.py
|   |   |       |   |-- ✅ utils/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ strptime.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ conf.py
|   |   |       |   |-- ✅ date.py
|   |   |       |   |-- ✅ date_parser.py
|   |   |       |   |-- ✅ freshness_date_parser.py
|   |   |       |   |-- ✅ parser.py
|   |   |       |   |-- ✅ timezone_parser.py
|   |   |       |   \-- ✅ timezones.py
|   |   |       |-- ✅ dateparser-1.2.2.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ AUTHORS.rst
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ dateparser_cli/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ cli.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ fasttext_manager.py
|   |   |       |   \-- ✅ utils.py
|   |   |       |-- ✅ dateparser_data/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   \-- ✅ settings.py
|   |   |       |-- ✅ dateparser_scripts/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ get_cldr_data.py
|   |   |       |   |-- ✅ order_languages.py
|   |   |       |   |-- ✅ update_supported_languages_and_locales.py
|   |   |       |   |-- ✅ utils.py
|   |   |       |   \-- ✅ write_complete_data.py
|   |   |       |-- ✅ dateutil/
|   |   |       |   |-- ✅ parser/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _parser.py
|   |   |       |   |   \-- ✅ isoparser.py
|   |   |       |   |-- ✅ tz/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _common.py
|   |   |       |   |   |-- ✅ _factories.py
|   |   |       |   |   |-- ✅ tz.py
|   |   |       |   |   \-- ✅ win.py
|   |   |       |   |-- ✅ zoneinfo/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ dateutil-zoneinfo.tar.gz
|   |   |       |   |   \-- ✅ rebuild.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _common.py
|   |   |       |   |-- ✅ _version.py
|   |   |       |   |-- ✅ easter.py
|   |   |       |   |-- ✅ relativedelta.py
|   |   |       |   |-- ✅ rrule.py
|   |   |       |   |-- ✅ tzwin.py
|   |   |       |   \-- ✅ utils.py
|   |   |       |-- ✅ dotenv/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __main__.py
|   |   |       |   |-- ✅ cli.py
|   |   |       |   |-- ✅ ipython.py
|   |   |       |   |-- ✅ main.py
|   |   |       |   |-- ✅ parser.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ variables.py
|   |   |       |   \-- ✅ version.py
|   |   |       |-- ✅ fastapi/
|   |   |       |   |-- ✅ dependencies/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ models.py
|   |   |       |   |   \-- ✅ utils.py
|   |   |       |   |-- ✅ middleware/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ cors.py
|   |   |       |   |   |-- ✅ gzip.py
|   |   |       |   |   |-- ✅ httpsredirect.py
|   |   |       |   |   |-- ✅ trustedhost.py
|   |   |       |   |   \-- ✅ wsgi.py
|   |   |       |   |-- ✅ openapi/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ constants.py
|   |   |       |   |   |-- ✅ docs.py
|   |   |       |   |   |-- ✅ models.py
|   |   |       |   |   \-- ✅ utils.py
|   |   |       |   |-- ✅ security/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ api_key.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ http.py
|   |   |       |   |   |-- ✅ oauth2.py
|   |   |       |   |   |-- ✅ open_id_connect_url.py
|   |   |       |   |   \-- ✅ utils.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __main__.py
|   |   |       |   |-- ✅ _compat.py
|   |   |       |   |-- ✅ applications.py
|   |   |       |   |-- ✅ background.py
|   |   |       |   |-- ✅ cli.py
|   |   |       |   |-- ✅ concurrency.py
|   |   |       |   |-- ✅ datastructures.py
|   |   |       |   |-- ✅ encoders.py
|   |   |       |   |-- ✅ exception_handlers.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ logger.py
|   |   |       |   |-- ✅ param_functions.py
|   |   |       |   |-- ✅ params.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ requests.py
|   |   |       |   |-- ✅ responses.py
|   |   |       |   |-- ✅ routing.py
|   |   |       |   |-- ✅ staticfiles.py
|   |   |       |   |-- ✅ templating.py
|   |   |       |   |-- ✅ testclient.py
|   |   |       |   |-- ✅ types.py
|   |   |       |   |-- ✅ utils.py
|   |   |       |   \-- ✅ websockets.py
|   |   |       |-- ✅ fastapi-0.117.1.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ fontTools/
|   |   |       |   |-- ✅ cffLib/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ CFF2ToCFF.py
|   |   |       |   |   |-- ✅ CFFToCFF2.py
|   |   |       |   |   |-- ✅ specializer.py
|   |   |       |   |   |-- ✅ transforms.py
|   |   |       |   |   \-- ✅ width.py
|   |   |       |   |-- ✅ colorLib/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ builder.py
|   |   |       |   |   |-- ✅ errors.py
|   |   |       |   |   |-- ✅ geometry.py
|   |   |       |   |   |-- ✅ table_builder.py
|   |   |       |   |   \-- ✅ unbuilder.py
|   |   |       |   |-- ✅ config/
|   |   |       |   |   \-- ✅ __init__.py
|   |   |       |   |-- ✅ cu2qu/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __main__.py
|   |   |       |   |   |-- ✅ benchmark.py
|   |   |       |   |   |-- ✅ cli.py
|   |   |       |   |   |-- ✅ cu2qu.c
|   |   |       |   |   |-- ✅ cu2qu.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ cu2qu.py
|   |   |       |   |   |-- ✅ errors.py
|   |   |       |   |   \-- ✅ ufo.py
|   |   |       |   |-- ✅ designspaceLib/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __main__.py
|   |   |       |   |   |-- ✅ split.py
|   |   |       |   |   |-- ✅ statNames.py
|   |   |       |   |   \-- ✅ types.py
|   |   |       |   |-- ✅ encodings/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ codecs.py
|   |   |       |   |   |-- ✅ MacRoman.py
|   |   |       |   |   \-- ✅ StandardEncoding.py
|   |   |       |   |-- ✅ feaLib/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __main__.py
|   |   |       |   |   |-- ✅ ast.py
|   |   |       |   |   |-- ✅ builder.py
|   |   |       |   |   |-- ✅ error.py
|   |   |       |   |   |-- ✅ lexer.c
|   |   |       |   |   |-- ✅ lexer.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ lexer.py
|   |   |       |   |   |-- ✅ location.py
|   |   |       |   |   |-- ✅ lookupDebugInfo.py
|   |   |       |   |   |-- ✅ parser.py
|   |   |       |   |   \-- ✅ variableScalar.py
|   |   |       |   |-- ✅ merge/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __main__.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ cmap.py
|   |   |       |   |   |-- ✅ layout.py
|   |   |       |   |   |-- ✅ options.py
|   |   |       |   |   |-- ✅ tables.py
|   |   |       |   |   |-- ✅ unicode.py
|   |   |       |   |   \-- ✅ util.py
|   |   |       |   |-- ✅ misc/
|   |   |       |   |   |-- ✅ filesystem/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _base.py
|   |   |       |   |   |   |-- ✅ _copy.py
|   |   |       |   |   |   |-- ✅ _errors.py
|   |   |       |   |   |   |-- ✅ _info.py
|   |   |       |   |   |   |-- ✅ _osfs.py
|   |   |       |   |   |   |-- ✅ _path.py
|   |   |       |   |   |   |-- ✅ _subfs.py
|   |   |       |   |   |   |-- ✅ _tempfs.py
|   |   |       |   |   |   |-- ✅ _tools.py
|   |   |       |   |   |   |-- ✅ _walk.py
|   |   |       |   |   |   \-- ✅ _zipfs.py
|   |   |       |   |   |-- ✅ plistlib/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ py.typed
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ arrayTools.py
|   |   |       |   |   |-- ✅ bezierTools.c
|   |   |       |   |   |-- ✅ bezierTools.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ bezierTools.py
|   |   |       |   |   |-- ✅ classifyTools.py
|   |   |       |   |   |-- ✅ cliTools.py
|   |   |       |   |   |-- ✅ configTools.py
|   |   |       |   |   |-- ✅ cython.py
|   |   |       |   |   |-- ✅ dictTools.py
|   |   |       |   |   |-- ✅ eexec.py
|   |   |       |   |   |-- ✅ encodingTools.py
|   |   |       |   |   |-- ✅ enumTools.py
|   |   |       |   |   |-- ✅ etree.py
|   |   |       |   |   |-- ✅ filenames.py
|   |   |       |   |   |-- ✅ fixedTools.py
|   |   |       |   |   |-- ✅ intTools.py
|   |   |       |   |   |-- ✅ iterTools.py
|   |   |       |   |   |-- ✅ lazyTools.py
|   |   |       |   |   |-- ✅ loggingTools.py
|   |   |       |   |   |-- ✅ macCreatorType.py
|   |   |       |   |   |-- ✅ macRes.py
|   |   |       |   |   |-- ✅ psCharStrings.py
|   |   |       |   |   |-- ✅ psLib.py
|   |   |       |   |   |-- ✅ psOperators.py
|   |   |       |   |   |-- ✅ py23.py
|   |   |       |   |   |-- ✅ roundTools.py
|   |   |       |   |   |-- ✅ sstruct.py
|   |   |       |   |   |-- ✅ symfont.py
|   |   |       |   |   |-- ✅ testTools.py
|   |   |       |   |   |-- ✅ textTools.py
|   |   |       |   |   |-- ✅ timeTools.py
|   |   |       |   |   |-- ✅ transform.py
|   |   |       |   |   |-- ✅ treeTools.py
|   |   |       |   |   |-- ✅ vector.py
|   |   |       |   |   |-- ✅ visitor.py
|   |   |       |   |   |-- ✅ xmlReader.py
|   |   |       |   |   \-- ✅ xmlWriter.py
|   |   |       |   |-- ✅ mtiLib/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ __main__.py
|   |   |       |   |-- ✅ otlLib/
|   |   |       |   |   |-- ✅ optimize/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __main__.py
|   |   |       |   |   |   \-- ✅ gpos.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ builder.py
|   |   |       |   |   |-- ✅ error.py
|   |   |       |   |   \-- ✅ maxContextCalc.py
|   |   |       |   |-- ✅ pens/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ areaPen.py
|   |   |       |   |   |-- ✅ basePen.py
|   |   |       |   |   |-- ✅ boundsPen.py
|   |   |       |   |   |-- ✅ cairoPen.py
|   |   |       |   |   |-- ✅ cocoaPen.py
|   |   |       |   |   |-- ✅ cu2quPen.py
|   |   |       |   |   |-- ✅ explicitClosingLinePen.py
|   |   |       |   |   |-- ✅ filterPen.py
|   |   |       |   |   |-- ✅ freetypePen.py
|   |   |       |   |   |-- ✅ hashPointPen.py
|   |   |       |   |   |-- ✅ momentsPen.c
|   |   |       |   |   |-- ✅ momentsPen.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ momentsPen.py
|   |   |       |   |   |-- ✅ perimeterPen.py
|   |   |       |   |   |-- ✅ pointInsidePen.py
|   |   |       |   |   |-- ✅ pointPen.py
|   |   |       |   |   |-- ✅ qtPen.py
|   |   |       |   |   |-- ✅ qu2cuPen.py
|   |   |       |   |   |-- ✅ quartzPen.py
|   |   |       |   |   |-- ✅ recordingPen.py
|   |   |       |   |   |-- ✅ reportLabPen.py
|   |   |       |   |   |-- ✅ reverseContourPen.py
|   |   |       |   |   |-- ✅ roundingPen.py
|   |   |       |   |   |-- ✅ statisticsPen.py
|   |   |       |   |   |-- ✅ svgPathPen.py
|   |   |       |   |   |-- ✅ t2CharStringPen.py
|   |   |       |   |   |-- ✅ teePen.py
|   |   |       |   |   |-- ✅ transformPen.py
|   |   |       |   |   |-- ✅ ttGlyphPen.py
|   |   |       |   |   \-- ✅ wxPen.py
|   |   |       |   |-- ✅ qu2cu/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __main__.py
|   |   |       |   |   |-- ✅ benchmark.py
|   |   |       |   |   |-- ✅ cli.py
|   |   |       |   |   |-- ✅ qu2cu.c
|   |   |       |   |   |-- ✅ qu2cu.cp312-win_amd64.pyd
|   |   |       |   |   \-- ✅ qu2cu.py
|   |   |       |   |-- ✅ subset/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __main__.py
|   |   |       |   |   |-- ✅ cff.py
|   |   |       |   |   |-- ✅ svg.py
|   |   |       |   |   \-- ✅ util.py
|   |   |       |   |-- ✅ svgLib/
|   |   |       |   |   |-- ✅ path/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ arc.py
|   |   |       |   |   |   |-- ✅ parser.py
|   |   |       |   |   |   \-- ✅ shapes.py
|   |   |       |   |   \-- ✅ __init__.py
|   |   |       |   |-- ✅ t1Lib/
|   |   |       |   |   \-- ✅ __init__.py
|   |   |       |   |-- ✅ ttLib/
|   |   |       |   |   |-- ✅ tables/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _a_n_k_r.py
|   |   |       |   |   |   |-- ✅ _a_v_a_r.py
|   |   |       |   |   |   |-- ✅ _b_s_l_n.py
|   |   |       |   |   |   |-- ✅ _c_i_d_g.py
|   |   |       |   |   |   |-- ✅ _c_m_a_p.py
|   |   |       |   |   |   |-- ✅ _c_v_a_r.py
|   |   |       |   |   |   |-- ✅ _c_v_t.py
|   |   |       |   |   |   |-- ✅ _f_e_a_t.py
|   |   |       |   |   |   |-- ✅ _f_p_g_m.py
|   |   |       |   |   |   |-- ✅ _f_v_a_r.py
|   |   |       |   |   |   |-- ✅ _g_a_s_p.py
|   |   |       |   |   |   |-- ✅ _g_c_i_d.py
|   |   |       |   |   |   |-- ✅ _g_l_y_f.py
|   |   |       |   |   |   |-- ✅ _g_v_a_r.py
|   |   |       |   |   |   |-- ✅ _h_d_m_x.py
|   |   |       |   |   |   |-- ✅ _h_e_a_d.py
|   |   |       |   |   |   |-- ✅ _h_h_e_a.py
|   |   |       |   |   |   |-- ✅ _h_m_t_x.py
|   |   |       |   |   |   |-- ✅ _k_e_r_n.py
|   |   |       |   |   |   |-- ✅ _l_c_a_r.py
|   |   |       |   |   |   |-- ✅ _l_o_c_a.py
|   |   |       |   |   |   |-- ✅ _l_t_a_g.py
|   |   |       |   |   |   |-- ✅ _m_a_x_p.py
|   |   |       |   |   |   |-- ✅ _m_e_t_a.py
|   |   |       |   |   |   |-- ✅ _m_o_r_t.py
|   |   |       |   |   |   |-- ✅ _m_o_r_x.py
|   |   |       |   |   |   |-- ✅ _n_a_m_e.py
|   |   |       |   |   |   |-- ✅ _o_p_b_d.py
|   |   |       |   |   |   |-- ✅ _p_o_s_t.py
|   |   |       |   |   |   |-- ✅ _p_r_e_p.py
|   |   |       |   |   |   |-- ✅ _p_r_o_p.py
|   |   |       |   |   |   |-- ✅ _s_b_i_x.py
|   |   |       |   |   |   |-- ✅ _t_r_a_k.py
|   |   |       |   |   |   |-- ✅ _v_h_e_a.py
|   |   |       |   |   |   |-- ✅ _v_m_t_x.py
|   |   |       |   |   |   |-- ✅ asciiTable.py
|   |   |       |   |   |   |-- ✅ B_A_S_E_.py
|   |   |       |   |   |   |-- ✅ BitmapGlyphMetrics.py
|   |   |       |   |   |   |-- ✅ C_B_D_T_.py
|   |   |       |   |   |   |-- ✅ C_B_L_C_.py
|   |   |       |   |   |   |-- ✅ C_F_F_.py
|   |   |       |   |   |   |-- ✅ C_F_F__2.py
|   |   |       |   |   |   |-- ✅ C_O_L_R_.py
|   |   |       |   |   |   |-- ✅ C_P_A_L_.py
|   |   |       |   |   |   |-- ✅ D__e_b_g.py
|   |   |       |   |   |   |-- ✅ D_S_I_G_.py
|   |   |       |   |   |   |-- ✅ DefaultTable.py
|   |   |       |   |   |   |-- ✅ E_B_D_T_.py
|   |   |       |   |   |   |-- ✅ E_B_L_C_.py
|   |   |       |   |   |   |-- ✅ F__e_a_t.py
|   |   |       |   |   |   |-- ✅ F_F_T_M_.py
|   |   |       |   |   |   |-- ✅ G__l_a_t.py
|   |   |       |   |   |   |-- ✅ G__l_o_c.py
|   |   |       |   |   |   |-- ✅ G_D_E_F_.py
|   |   |       |   |   |   |-- ✅ G_M_A_P_.py
|   |   |       |   |   |   |-- ✅ G_P_K_G_.py
|   |   |       |   |   |   |-- ✅ G_P_O_S_.py
|   |   |       |   |   |   |-- ✅ G_S_U_B_.py
|   |   |       |   |   |   |-- ✅ G_V_A_R_.py
|   |   |       |   |   |   |-- ✅ grUtils.py
|   |   |       |   |   |   |-- ✅ H_V_A_R_.py
|   |   |       |   |   |   |-- ✅ J_S_T_F_.py
|   |   |       |   |   |   |-- ✅ L_T_S_H_.py
|   |   |       |   |   |   |-- ✅ M_A_T_H_.py
|   |   |       |   |   |   |-- ✅ M_E_T_A_.py
|   |   |       |   |   |   |-- ✅ M_V_A_R_.py
|   |   |       |   |   |   |-- ✅ O_S_2f_2.py
|   |   |       |   |   |   |-- ✅ otBase.py
|   |   |       |   |   |   |-- ✅ otConverters.py
|   |   |       |   |   |   |-- ✅ otData.py
|   |   |       |   |   |   |-- ✅ otTables.py
|   |   |       |   |   |   |-- ✅ otTraverse.py
|   |   |       |   |   |   |-- ✅ S__i_l_f.py
|   |   |       |   |   |   |-- ✅ S__i_l_l.py
|   |   |       |   |   |   |-- ✅ S_I_N_G_.py
|   |   |       |   |   |   |-- ✅ S_T_A_T_.py
|   |   |       |   |   |   |-- ✅ S_V_G_.py
|   |   |       |   |   |   |-- ✅ sbixGlyph.py
|   |   |       |   |   |   |-- ✅ sbixStrike.py
|   |   |       |   |   |   |-- ✅ T_S_I__0.py
|   |   |       |   |   |   |-- ✅ T_S_I__1.py
|   |   |       |   |   |   |-- ✅ T_S_I__2.py
|   |   |       |   |   |   |-- ✅ T_S_I__3.py
|   |   |       |   |   |   |-- ✅ T_S_I__5.py
|   |   |       |   |   |   |-- ✅ T_S_I_B_.py
|   |   |       |   |   |   |-- ✅ T_S_I_C_.py
|   |   |       |   |   |   |-- ✅ T_S_I_D_.py
|   |   |       |   |   |   |-- ✅ T_S_I_J_.py
|   |   |       |   |   |   |-- ✅ T_S_I_P_.py
|   |   |       |   |   |   |-- ✅ T_S_I_S_.py
|   |   |       |   |   |   |-- ✅ T_S_I_V_.py
|   |   |       |   |   |   |-- ✅ T_T_F_A_.py
|   |   |       |   |   |   |-- ✅ table_API_readme.txt
|   |   |       |   |   |   |-- ✅ ttProgram.py
|   |   |       |   |   |   |-- ✅ TupleVariation.py
|   |   |       |   |   |   |-- ✅ V_A_R_C_.py
|   |   |       |   |   |   |-- ✅ V_D_M_X_.py
|   |   |       |   |   |   |-- ✅ V_O_R_G_.py
|   |   |       |   |   |   \-- ✅ V_V_A_R_.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __main__.py
|   |   |       |   |   |-- ✅ macUtils.py
|   |   |       |   |   |-- ✅ removeOverlaps.py
|   |   |       |   |   |-- ✅ reorderGlyphs.py
|   |   |       |   |   |-- ✅ scaleUpem.py
|   |   |       |   |   |-- ✅ sfnt.py
|   |   |       |   |   |-- ✅ standardGlyphOrder.py
|   |   |       |   |   |-- ✅ ttCollection.py
|   |   |       |   |   |-- ✅ ttFont.py
|   |   |       |   |   |-- ✅ ttGlyphSet.py
|   |   |       |   |   |-- ✅ ttVisitor.py
|   |   |       |   |   \-- ✅ woff2.py
|   |   |       |   |-- ✅ ufoLib/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ converters.py
|   |   |       |   |   |-- ✅ errors.py
|   |   |       |   |   |-- ✅ etree.py
|   |   |       |   |   |-- ✅ filenames.py
|   |   |       |   |   |-- ✅ glifLib.py
|   |   |       |   |   |-- ✅ kerning.py
|   |   |       |   |   |-- ✅ plistlib.py
|   |   |       |   |   |-- ✅ pointPen.py
|   |   |       |   |   |-- ✅ utils.py
|   |   |       |   |   \-- ✅ validators.py
|   |   |       |   |-- ✅ unicodedata/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ Blocks.py
|   |   |       |   |   |-- ✅ Mirrored.py
|   |   |       |   |   |-- ✅ OTTags.py
|   |   |       |   |   |-- ✅ ScriptExtensions.py
|   |   |       |   |   \-- ✅ Scripts.py
|   |   |       |   |-- ✅ varLib/
|   |   |       |   |   |-- ✅ avar/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __main__.py
|   |   |       |   |   |   |-- ✅ build.py
|   |   |       |   |   |   |-- ✅ map.py
|   |   |       |   |   |   |-- ✅ plan.py
|   |   |       |   |   |   \-- ✅ unbuild.py
|   |   |       |   |   |-- ✅ instancer/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __main__.py
|   |   |       |   |   |   |-- ✅ featureVars.py
|   |   |       |   |   |   |-- ✅ names.py
|   |   |       |   |   |   \-- ✅ solver.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __main__.py
|   |   |       |   |   |-- ✅ avarPlanner.py
|   |   |       |   |   |-- ✅ builder.py
|   |   |       |   |   |-- ✅ cff.py
|   |   |       |   |   |-- ✅ errors.py
|   |   |       |   |   |-- ✅ featureVars.py
|   |   |       |   |   |-- ✅ hvar.py
|   |   |       |   |   |-- ✅ interpolatable.py
|   |   |       |   |   |-- ✅ interpolatableHelpers.py
|   |   |       |   |   |-- ✅ interpolatablePlot.py
|   |   |       |   |   |-- ✅ interpolatableTestContourOrder.py
|   |   |       |   |   |-- ✅ interpolatableTestStartingPoint.py
|   |   |       |   |   |-- ✅ interpolate_layout.py
|   |   |       |   |   |-- ✅ iup.c
|   |   |       |   |   |-- ✅ iup.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ iup.py
|   |   |       |   |   |-- ✅ merger.py
|   |   |       |   |   |-- ✅ models.py
|   |   |       |   |   |-- ✅ multiVarStore.py
|   |   |       |   |   |-- ✅ mutator.py
|   |   |       |   |   |-- ✅ mvar.py
|   |   |       |   |   |-- ✅ plot.py
|   |   |       |   |   |-- ✅ stat.py
|   |   |       |   |   \-- ✅ varStore.py
|   |   |       |   |-- ✅ voltLib/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __main__.py
|   |   |       |   |   |-- ✅ ast.py
|   |   |       |   |   |-- ✅ error.py
|   |   |       |   |   |-- ✅ lexer.py
|   |   |       |   |   |-- ✅ parser.py
|   |   |       |   |   \-- ✅ voltToFea.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __main__.py
|   |   |       |   |-- ✅ afmLib.py
|   |   |       |   |-- ✅ agl.py
|   |   |       |   |-- ✅ annotations.py
|   |   |       |   |-- ✅ fontBuilder.py
|   |   |       |   |-- ✅ help.py
|   |   |       |   |-- ✅ tfmLib.py
|   |   |       |   |-- ✅ ttx.py
|   |   |       |   \-- ✅ unicode.py
|   |   |       |-- ✅ fonttools-4.60.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ LICENSE
|   |   |       |   |   \-- ✅ LICENSE.external
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ frozenlist/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __init__.pyi
|   |   |       |   |-- ✅ _frozenlist.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _frozenlist.pyx
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ frozenlist-1.7.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ greenlet/
|   |   |       |   |-- ✅ platform/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ setup_switch_x64_masm.cmd
|   |   |       |   |   |-- ✅ switch_aarch64_gcc.h
|   |   |       |   |   |-- ✅ switch_alpha_unix.h
|   |   |       |   |   |-- ✅ switch_amd64_unix.h
|   |   |       |   |   |-- ✅ switch_arm32_gcc.h
|   |   |       |   |   |-- ✅ switch_arm32_ios.h
|   |   |       |   |   |-- ✅ switch_arm64_masm.asm
|   |   |       |   |   |-- ✅ switch_arm64_masm.obj
|   |   |       |   |   |-- ✅ switch_arm64_msvc.h
|   |   |       |   |   |-- ✅ switch_csky_gcc.h
|   |   |       |   |   |-- ✅ switch_loongarch64_linux.h
|   |   |       |   |   |-- ✅ switch_m68k_gcc.h
|   |   |       |   |   |-- ✅ switch_mips_unix.h
|   |   |       |   |   |-- ✅ switch_ppc64_aix.h
|   |   |       |   |   |-- ✅ switch_ppc64_linux.h
|   |   |       |   |   |-- ✅ switch_ppc_aix.h
|   |   |       |   |   |-- ✅ switch_ppc_linux.h
|   |   |       |   |   |-- ✅ switch_ppc_macosx.h
|   |   |       |   |   |-- ✅ switch_ppc_unix.h
|   |   |       |   |   |-- ✅ switch_riscv_unix.h
|   |   |       |   |   |-- ✅ switch_s390_unix.h
|   |   |       |   |   |-- ✅ switch_sh_gcc.h
|   |   |       |   |   |-- ✅ switch_sparc_sun_gcc.h
|   |   |       |   |   |-- ✅ switch_x32_unix.h
|   |   |       |   |   |-- ✅ switch_x64_masm.asm
|   |   |       |   |   |-- ✅ switch_x64_masm.obj
|   |   |       |   |   |-- ✅ switch_x64_msvc.h
|   |   |       |   |   |-- ✅ switch_x86_msvc.h
|   |   |       |   |   \-- ✅ switch_x86_unix.h
|   |   |       |   |-- ✅ tests/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _test_extension.c
|   |   |       |   |   |-- ✅ _test_extension.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _test_extension_cpp.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _test_extension_cpp.cpp
|   |   |       |   |   |-- ✅ fail_clearing_run_switches.py
|   |   |       |   |   |-- ✅ fail_cpp_exception.py
|   |   |       |   |   |-- ✅ fail_initialstub_already_started.py
|   |   |       |   |   |-- ✅ fail_slp_switch.py
|   |   |       |   |   |-- ✅ fail_switch_three_greenlets.py
|   |   |       |   |   |-- ✅ fail_switch_three_greenlets2.py
|   |   |       |   |   |-- ✅ fail_switch_two_greenlets.py
|   |   |       |   |   |-- ✅ leakcheck.py
|   |   |       |   |   |-- ✅ test_contextvars.py
|   |   |       |   |   |-- ✅ test_cpp.py
|   |   |       |   |   |-- ✅ test_extension_interface.py
|   |   |       |   |   |-- ✅ test_gc.py
|   |   |       |   |   |-- ✅ test_generator.py
|   |   |       |   |   |-- ✅ test_generator_nested.py
|   |   |       |   |   |-- ✅ test_greenlet.py
|   |   |       |   |   |-- ✅ test_greenlet_trash.py
|   |   |       |   |   |-- ✅ test_leaks.py
|   |   |       |   |   |-- ✅ test_stack_saved.py
|   |   |       |   |   |-- ✅ test_throw.py
|   |   |       |   |   |-- ✅ test_tracing.py
|   |   |       |   |   |-- ✅ test_version.py
|   |   |       |   |   \-- ✅ test_weakref.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _greenlet.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ CObjects.cpp
|   |   |       |   |-- ✅ greenlet.cpp
|   |   |       |   |-- ✅ greenlet.h
|   |   |       |   |-- ✅ greenlet_allocator.hpp
|   |   |       |   |-- ✅ greenlet_compiler_compat.hpp
|   |   |       |   |-- ✅ greenlet_cpython_compat.hpp
|   |   |       |   |-- ✅ greenlet_exceptions.hpp
|   |   |       |   |-- ✅ greenlet_internal.hpp
|   |   |       |   |-- ✅ greenlet_msvc_compat.hpp
|   |   |       |   |-- ✅ greenlet_refs.hpp
|   |   |       |   |-- ✅ greenlet_slp_switch.hpp
|   |   |       |   |-- ✅ greenlet_thread_support.hpp
|   |   |       |   |-- ✅ PyGreenlet.cpp
|   |   |       |   |-- ✅ PyGreenlet.hpp
|   |   |       |   |-- ✅ PyGreenletUnswitchable.cpp
|   |   |       |   |-- ✅ PyModule.cpp
|   |   |       |   |-- ✅ slp_platformselect.h
|   |   |       |   |-- ✅ TBrokenGreenlet.cpp
|   |   |       |   |-- ✅ TExceptionState.cpp
|   |   |       |   |-- ✅ TGreenlet.cpp
|   |   |       |   |-- ✅ TGreenlet.hpp
|   |   |       |   |-- ✅ TGreenletGlobals.cpp
|   |   |       |   |-- ✅ TMainGreenlet.cpp
|   |   |       |   |-- ✅ TPythonState.cpp
|   |   |       |   |-- ✅ TStackState.cpp
|   |   |       |   |-- ✅ TThreadState.hpp
|   |   |       |   |-- ✅ TThreadStateCreator.hpp
|   |   |       |   |-- ✅ TThreadStateDestroy.cpp
|   |   |       |   \-- ✅ TUserGreenlet.cpp
|   |   |       |-- ✅ greenlet-3.2.4.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ LICENSE
|   |   |       |   |   \-- ✅ LICENSE.PSF
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ h11/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _abnf.py
|   |   |       |   |-- ✅ _connection.py
|   |   |       |   |-- ✅ _events.py
|   |   |       |   |-- ✅ _headers.py
|   |   |       |   |-- ✅ _readers.py
|   |   |       |   |-- ✅ _receivebuffer.py
|   |   |       |   |-- ✅ _state.py
|   |   |       |   |-- ✅ _util.py
|   |   |       |   |-- ✅ _version.py
|   |   |       |   |-- ✅ _writers.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ h11-0.16.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ httpcore/
|   |   |       |   |-- ✅ _async/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ connection.py
|   |   |       |   |   |-- ✅ connection_pool.py
|   |   |       |   |   |-- ✅ http11.py
|   |   |       |   |   |-- ✅ http2.py
|   |   |       |   |   |-- ✅ http_proxy.py
|   |   |       |   |   |-- ✅ interfaces.py
|   |   |       |   |   \-- ✅ socks_proxy.py
|   |   |       |   |-- ✅ _backends/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ anyio.py
|   |   |       |   |   |-- ✅ auto.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ mock.py
|   |   |       |   |   |-- ✅ sync.py
|   |   |       |   |   \-- ✅ trio.py
|   |   |       |   |-- ✅ _sync/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ connection.py
|   |   |       |   |   |-- ✅ connection_pool.py
|   |   |       |   |   |-- ✅ http11.py
|   |   |       |   |   |-- ✅ http2.py
|   |   |       |   |   |-- ✅ http_proxy.py
|   |   |       |   |   |-- ✅ interfaces.py
|   |   |       |   |   \-- ✅ socks_proxy.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _api.py
|   |   |       |   |-- ✅ _exceptions.py
|   |   |       |   |-- ✅ _models.py
|   |   |       |   |-- ✅ _ssl.py
|   |   |       |   |-- ✅ _synchronization.py
|   |   |       |   |-- ✅ _trace.py
|   |   |       |   |-- ✅ _utils.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ httpcore-1.0.9.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE.md
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ httpx/
|   |   |       |   |-- ✅ _transports/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ asgi.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ default.py
|   |   |       |   |   |-- ✅ mock.py
|   |   |       |   |   \-- ✅ wsgi.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __version__.py
|   |   |       |   |-- ✅ _api.py
|   |   |       |   |-- ✅ _auth.py
|   |   |       |   |-- ✅ _client.py
|   |   |       |   |-- ✅ _config.py
|   |   |       |   |-- ✅ _content.py
|   |   |       |   |-- ✅ _decoders.py
|   |   |       |   |-- ✅ _exceptions.py
|   |   |       |   |-- ✅ _main.py
|   |   |       |   |-- ✅ _models.py
|   |   |       |   |-- ✅ _multipart.py
|   |   |       |   |-- ✅ _status_codes.py
|   |   |       |   |-- ✅ _types.py
|   |   |       |   |-- ✅ _urlparse.py
|   |   |       |   |-- ✅ _urls.py
|   |   |       |   |-- ✅ _utils.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ httpx-0.28.1.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE.md
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ idna/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ codec.py
|   |   |       |   |-- ✅ compat.py
|   |   |       |   |-- ✅ core.py
|   |   |       |   |-- ✅ idnadata.py
|   |   |       |   |-- ✅ intranges.py
|   |   |       |   |-- ✅ package_data.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   \-- ✅ uts46data.py
|   |   |       |-- ✅ idna-3.10.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE.md
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ joblib/
|   |   |       |   |-- ✅ externals/
|   |   |       |   |   |-- ✅ cloudpickle/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ cloudpickle.py
|   |   |       |   |   |   \-- ✅ cloudpickle_fast.py
|   |   |       |   |   |-- ✅ loky/
|   |   |       |   |   |   |-- ✅ backend/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _posix_reduction.py
|   |   |       |   |   |   |   |-- ✅ _win_reduction.py
|   |   |       |   |   |   |   |-- ✅ context.py
|   |   |       |   |   |   |   |-- ✅ fork_exec.py
|   |   |       |   |   |   |   |-- ✅ popen_loky_posix.py
|   |   |       |   |   |   |   |-- ✅ popen_loky_win32.py
|   |   |       |   |   |   |   |-- ✅ process.py
|   |   |       |   |   |   |   |-- ✅ queues.py
|   |   |       |   |   |   |   |-- ✅ reduction.py
|   |   |       |   |   |   |   |-- ✅ resource_tracker.py
|   |   |       |   |   |   |   |-- ✅ spawn.py
|   |   |       |   |   |   |   |-- ✅ synchronize.py
|   |   |       |   |   |   |   \-- ✅ utils.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _base.py
|   |   |       |   |   |   |-- ✅ cloudpickle_wrapper.py
|   |   |       |   |   |   |-- ✅ initializers.py
|   |   |       |   |   |   |-- ✅ process_executor.py
|   |   |       |   |   |   \-- ✅ reusable_executor.py
|   |   |       |   |   \-- ✅ __init__.py
|   |   |       |   |-- ✅ test/
|   |   |       |   |   |-- ✅ data/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ create_numpy_pickle.py
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_compressed_pickle_py27_np16.gz
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_compressed_pickle_py27_np17.gz
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_compressed_pickle_py33_np18.gz
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_compressed_pickle_py34_np19.gz
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_compressed_pickle_py35_np19.gz
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py27_np17.pkl
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py27_np17.pkl.bz2
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py27_np17.pkl.gzip
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py27_np17.pkl.lzma
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py27_np17.pkl.xz
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py33_np18.pkl
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py33_np18.pkl.bz2
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py33_np18.pkl.gzip
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py33_np18.pkl.lzma
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py33_np18.pkl.xz
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py34_np19.pkl
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py34_np19.pkl.bz2
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py34_np19.pkl.gzip
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py34_np19.pkl.lzma
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py34_np19.pkl.xz
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py35_np19.pkl
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py35_np19.pkl.bz2
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py35_np19.pkl.gzip
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py35_np19.pkl.lzma
|   |   |       |   |   |   |-- ✅ joblib_0.10.0_pickle_py35_np19.pkl.xz
|   |   |       |   |   |   |-- ✅ joblib_0.11.0_compressed_pickle_py36_np111.gz
|   |   |       |   |   |   |-- ✅ joblib_0.11.0_pickle_py36_np111.pkl
|   |   |       |   |   |   |-- ✅ joblib_0.11.0_pickle_py36_np111.pkl.bz2
|   |   |       |   |   |   |-- ✅ joblib_0.11.0_pickle_py36_np111.pkl.gzip
|   |   |       |   |   |   |-- ✅ joblib_0.11.0_pickle_py36_np111.pkl.lzma
|   |   |       |   |   |   |-- ✅ joblib_0.11.0_pickle_py36_np111.pkl.xz
|   |   |       |   |   |   |-- ✅ joblib_0.8.4_compressed_pickle_py27_np17.gz
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_compressed_pickle_py27_np16.gz
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_compressed_pickle_py27_np17.gz
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_compressed_pickle_py34_np19.gz
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_compressed_pickle_py35_np19.gz
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py27_np16.pkl
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py27_np16.pkl_01.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py27_np16.pkl_02.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py27_np16.pkl_03.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py27_np16.pkl_04.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py27_np17.pkl
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py27_np17.pkl_01.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py27_np17.pkl_02.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py27_np17.pkl_03.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py27_np17.pkl_04.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py33_np18.pkl
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py33_np18.pkl_01.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py33_np18.pkl_02.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py33_np18.pkl_03.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py33_np18.pkl_04.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py34_np19.pkl
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py34_np19.pkl_01.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py34_np19.pkl_02.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py34_np19.pkl_03.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py34_np19.pkl_04.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py35_np19.pkl
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py35_np19.pkl_01.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py35_np19.pkl_02.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py35_np19.pkl_03.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.2_pickle_py35_np19.pkl_04.npy
|   |   |       |   |   |   |-- ✅ joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz
|   |   |       |   |   |   |-- ✅ joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz_01.npy.z
|   |   |       |   |   |   |-- ✅ joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz_02.npy.z
|   |   |       |   |   |   \-- ✅ joblib_0.9.4.dev0_compressed_cache_size_pickle_py35_np19.gz_03.npy.z
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ common.py
|   |   |       |   |   |-- ✅ test_backports.py
|   |   |       |   |   |-- ✅ test_cloudpickle_wrapper.py
|   |   |       |   |   |-- ✅ test_config.py
|   |   |       |   |   |-- ✅ test_dask.py
|   |   |       |   |   |-- ✅ test_disk.py
|   |   |       |   |   |-- ✅ test_func_inspect.py
|   |   |       |   |   |-- ✅ test_func_inspect_special_encoding.py
|   |   |       |   |   |-- ✅ test_hashing.py
|   |   |       |   |   |-- ✅ test_init.py
|   |   |       |   |   |-- ✅ test_logger.py
|   |   |       |   |   |-- ✅ test_memmapping.py
|   |   |       |   |   |-- ✅ test_memory.py
|   |   |       |   |   |-- ✅ test_memory_async.py
|   |   |       |   |   |-- ✅ test_missing_multiprocessing.py
|   |   |       |   |   |-- ✅ test_module.py
|   |   |       |   |   |-- ✅ test_numpy_pickle.py
|   |   |       |   |   |-- ✅ test_numpy_pickle_compat.py
|   |   |       |   |   |-- ✅ test_numpy_pickle_utils.py
|   |   |       |   |   |-- ✅ test_parallel.py
|   |   |       |   |   |-- ✅ test_store_backends.py
|   |   |       |   |   |-- ✅ test_testing.py
|   |   |       |   |   |-- ✅ test_utils.py
|   |   |       |   |   \-- ✅ testutils.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _cloudpickle_wrapper.py
|   |   |       |   |-- ✅ _dask.py
|   |   |       |   |-- ✅ _memmapping_reducer.py
|   |   |       |   |-- ✅ _multiprocessing_helpers.py
|   |   |       |   |-- ✅ _parallel_backends.py
|   |   |       |   |-- ✅ _store_backends.py
|   |   |       |   |-- ✅ _utils.py
|   |   |       |   |-- ✅ backports.py
|   |   |       |   |-- ✅ compressor.py
|   |   |       |   |-- ✅ disk.py
|   |   |       |   |-- ✅ executor.py
|   |   |       |   |-- ✅ func_inspect.py
|   |   |       |   |-- ✅ hashing.py
|   |   |       |   |-- ✅ logger.py
|   |   |       |   |-- ✅ memory.py
|   |   |       |   |-- ✅ numpy_pickle.py
|   |   |       |   |-- ✅ numpy_pickle_compat.py
|   |   |       |   |-- ✅ numpy_pickle_utils.py
|   |   |       |   |-- ✅ parallel.py
|   |   |       |   |-- ✅ pool.py
|   |   |       |   \-- ✅ testing.py
|   |   |       |-- ✅ joblib-1.5.2.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ kiwisolver/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _cext.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _cext.pyi
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ kiwisolver-1.4.9.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ matplotlib/
|   |   |       |   |-- ✅ _api/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ deprecation.py
|   |   |       |   |   \-- ✅ deprecation.pyi
|   |   |       |   |-- ✅ axes/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _axes.py
|   |   |       |   |   |-- ✅ _axes.pyi
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _base.pyi
|   |   |       |   |   |-- ✅ _secondary_axes.py
|   |   |       |   |   \-- ✅ _secondary_axes.pyi
|   |   |       |   |-- ✅ backends/
|   |   |       |   |   |-- ✅ qt_editor/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _formlayout.py
|   |   |       |   |   |   \-- ✅ figureoptions.py
|   |   |       |   |   |-- ✅ web_backend/
|   |   |       |   |   |   |-- ✅ css/
|   |   |       |   |   |   |   |-- ✅ boilerplate.css
|   |   |       |   |   |   |   |-- ✅ fbm.css
|   |   |       |   |   |   |   |-- ✅ mpl.css
|   |   |       |   |   |   |   \-- ✅ page.css
|   |   |       |   |   |   |-- ✅ js/
|   |   |       |   |   |   |   |-- ✅ mpl.js
|   |   |       |   |   |   |   |-- ✅ mpl_tornado.js
|   |   |       |   |   |   |   \-- ✅ nbagg_mpl.js
|   |   |       |   |   |   |-- ✅ all_figures.html
|   |   |       |   |   |   |-- ✅ ipython_inline_figure.html
|   |   |       |   |   |   \-- ✅ single_figure.html
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _backend_agg.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _backend_agg.pyi
|   |   |       |   |   |-- ✅ _backend_gtk.py
|   |   |       |   |   |-- ✅ _backend_pdf_ps.py
|   |   |       |   |   |-- ✅ _backend_tk.py
|   |   |       |   |   |-- ✅ _macosx.pyi
|   |   |       |   |   |-- ✅ _tkagg.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _tkagg.pyi
|   |   |       |   |   |-- ✅ backend_agg.py
|   |   |       |   |   |-- ✅ backend_cairo.py
|   |   |       |   |   |-- ✅ backend_gtk3.py
|   |   |       |   |   |-- ✅ backend_gtk3agg.py
|   |   |       |   |   |-- ✅ backend_gtk3cairo.py
|   |   |       |   |   |-- ✅ backend_gtk4.py
|   |   |       |   |   |-- ✅ backend_gtk4agg.py
|   |   |       |   |   |-- ✅ backend_gtk4cairo.py
|   |   |       |   |   |-- ✅ backend_macosx.py
|   |   |       |   |   |-- ✅ backend_mixed.py
|   |   |       |   |   |-- ✅ backend_nbagg.py
|   |   |       |   |   |-- ✅ backend_pdf.py
|   |   |       |   |   |-- ✅ backend_pgf.py
|   |   |       |   |   |-- ✅ backend_ps.py
|   |   |       |   |   |-- ✅ backend_qt.py
|   |   |       |   |   |-- ✅ backend_qt5.py
|   |   |       |   |   |-- ✅ backend_qt5agg.py
|   |   |       |   |   |-- ✅ backend_qt5cairo.py
|   |   |       |   |   |-- ✅ backend_qtagg.py
|   |   |       |   |   |-- ✅ backend_qtcairo.py
|   |   |       |   |   |-- ✅ backend_svg.py
|   |   |       |   |   |-- ✅ backend_template.py
|   |   |       |   |   |-- ✅ backend_tkagg.py
|   |   |       |   |   |-- ✅ backend_tkcairo.py
|   |   |       |   |   |-- ✅ backend_webagg.py
|   |   |       |   |   |-- ✅ backend_webagg_core.py
|   |   |       |   |   |-- ✅ backend_wx.py
|   |   |       |   |   |-- ✅ backend_wxagg.py
|   |   |       |   |   |-- ✅ backend_wxcairo.py
|   |   |       |   |   |-- ✅ qt_compat.py
|   |   |       |   |   \-- ✅ registry.py
|   |   |       |   |-- ✅ mpl-data/
|   |   |       |   |   |-- ✅ fonts/
|   |   |       |   |   |   |-- ✅ afm/
|   |   |       |   |   |   |   |-- ✅ cmex10.afm
|   |   |       |   |   |   |   |-- ✅ cmmi10.afm
|   |   |       |   |   |   |   |-- ✅ cmr10.afm
|   |   |       |   |   |   |   |-- ✅ cmsy10.afm
|   |   |       |   |   |   |   |-- ✅ cmtt10.afm
|   |   |       |   |   |   |   |-- ✅ pagd8a.afm
|   |   |       |   |   |   |   |-- ✅ pagdo8a.afm
|   |   |       |   |   |   |   |-- ✅ pagk8a.afm
|   |   |       |   |   |   |   |-- ✅ pagko8a.afm
|   |   |       |   |   |   |   |-- ✅ pbkd8a.afm
|   |   |       |   |   |   |   |-- ✅ pbkdi8a.afm
|   |   |       |   |   |   |   |-- ✅ pbkl8a.afm
|   |   |       |   |   |   |   |-- ✅ pbkli8a.afm
|   |   |       |   |   |   |   |-- ✅ pcrb8a.afm
|   |   |       |   |   |   |   |-- ✅ pcrbo8a.afm
|   |   |       |   |   |   |   |-- ✅ pcrr8a.afm
|   |   |       |   |   |   |   |-- ✅ pcrro8a.afm
|   |   |       |   |   |   |   |-- ✅ phvb8a.afm
|   |   |       |   |   |   |   |-- ✅ phvb8an.afm
|   |   |       |   |   |   |   |-- ✅ phvbo8a.afm
|   |   |       |   |   |   |   |-- ✅ phvbo8an.afm
|   |   |       |   |   |   |   |-- ✅ phvl8a.afm
|   |   |       |   |   |   |   |-- ✅ phvlo8a.afm
|   |   |       |   |   |   |   |-- ✅ phvr8a.afm
|   |   |       |   |   |   |   |-- ✅ phvr8an.afm
|   |   |       |   |   |   |   |-- ✅ phvro8a.afm
|   |   |       |   |   |   |   |-- ✅ phvro8an.afm
|   |   |       |   |   |   |   |-- ✅ pncb8a.afm
|   |   |       |   |   |   |   |-- ✅ pncbi8a.afm
|   |   |       |   |   |   |   |-- ✅ pncr8a.afm
|   |   |       |   |   |   |   |-- ✅ pncri8a.afm
|   |   |       |   |   |   |   |-- ✅ pplb8a.afm
|   |   |       |   |   |   |   |-- ✅ pplbi8a.afm
|   |   |       |   |   |   |   |-- ✅ pplr8a.afm
|   |   |       |   |   |   |   |-- ✅ pplri8a.afm
|   |   |       |   |   |   |   |-- ✅ psyr.afm
|   |   |       |   |   |   |   |-- ✅ ptmb8a.afm
|   |   |       |   |   |   |   |-- ✅ ptmbi8a.afm
|   |   |       |   |   |   |   |-- ✅ ptmr8a.afm
|   |   |       |   |   |   |   |-- ✅ ptmri8a.afm
|   |   |       |   |   |   |   |-- ✅ putb8a.afm
|   |   |       |   |   |   |   |-- ✅ putbi8a.afm
|   |   |       |   |   |   |   |-- ✅ putr8a.afm
|   |   |       |   |   |   |   |-- ✅ putri8a.afm
|   |   |       |   |   |   |   |-- ✅ pzcmi8a.afm
|   |   |       |   |   |   |   \-- ✅ pzdr.afm
|   |   |       |   |   |   |-- ✅ pdfcorefonts/
|   |   |       |   |   |   |   |-- ✅ Courier-Bold.afm
|   |   |       |   |   |   |   |-- ✅ Courier-BoldOblique.afm
|   |   |       |   |   |   |   |-- ✅ Courier-Oblique.afm
|   |   |       |   |   |   |   |-- ✅ Courier.afm
|   |   |       |   |   |   |   |-- ✅ Helvetica-Bold.afm
|   |   |       |   |   |   |   |-- ✅ Helvetica-BoldOblique.afm
|   |   |       |   |   |   |   |-- ✅ Helvetica-Oblique.afm
|   |   |       |   |   |   |   |-- ✅ Helvetica.afm
|   |   |       |   |   |   |   |-- ✅ readme.txt
|   |   |       |   |   |   |   |-- ✅ Symbol.afm
|   |   |       |   |   |   |   |-- ✅ Times-Bold.afm
|   |   |       |   |   |   |   |-- ✅ Times-BoldItalic.afm
|   |   |       |   |   |   |   |-- ✅ Times-Italic.afm
|   |   |       |   |   |   |   |-- ✅ Times-Roman.afm
|   |   |       |   |   |   |   \-- ✅ ZapfDingbats.afm
|   |   |       |   |   |   \-- ✅ ttf/
|   |   |       |   |   |       |-- ✅ cmb10.ttf
|   |   |       |   |   |       |-- ✅ cmex10.ttf
|   |   |       |   |   |       |-- ✅ cmmi10.ttf
|   |   |       |   |   |       |-- ✅ cmr10.ttf
|   |   |       |   |   |       |-- ✅ cmss10.ttf
|   |   |       |   |   |       |-- ✅ cmsy10.ttf
|   |   |       |   |   |       |-- ✅ cmtt10.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSans-Bold.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSans-BoldOblique.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSans-Oblique.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSans.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSansDisplay.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSansMono-Bold.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSansMono-BoldOblique.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSansMono-Oblique.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSansMono.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSerif-Bold.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSerif-BoldItalic.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSerif-Italic.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSerif.ttf
|   |   |       |   |   |       |-- ✅ DejaVuSerifDisplay.ttf
|   |   |       |   |   |       |-- ✅ LICENSE_DEJAVU
|   |   |       |   |   |       |-- ✅ LICENSE_STIX
|   |   |       |   |   |       |-- ✅ STIXGeneral.ttf
|   |   |       |   |   |       |-- ✅ STIXGeneralBol.ttf
|   |   |       |   |   |       |-- ✅ STIXGeneralBolIta.ttf
|   |   |       |   |   |       |-- ✅ STIXGeneralItalic.ttf
|   |   |       |   |   |       |-- ✅ STIXNonUni.ttf
|   |   |       |   |   |       |-- ✅ STIXNonUniBol.ttf
|   |   |       |   |   |       |-- ✅ STIXNonUniBolIta.ttf
|   |   |       |   |   |       |-- ✅ STIXNonUniIta.ttf
|   |   |       |   |   |       |-- ✅ STIXSizFiveSymReg.ttf
|   |   |       |   |   |       |-- ✅ STIXSizFourSymBol.ttf
|   |   |       |   |   |       |-- ✅ STIXSizFourSymReg.ttf
|   |   |       |   |   |       |-- ✅ STIXSizOneSymBol.ttf
|   |   |       |   |   |       |-- ✅ STIXSizOneSymReg.ttf
|   |   |       |   |   |       |-- ✅ STIXSizThreeSymBol.ttf
|   |   |       |   |   |       |-- ✅ STIXSizThreeSymReg.ttf
|   |   |       |   |   |       |-- ✅ STIXSizTwoSymBol.ttf
|   |   |       |   |   |       \-- ✅ STIXSizTwoSymReg.ttf
|   |   |       |   |   |-- ✅ images/
|   |   |       |   |   |   |-- ✅ back-symbolic.svg
|   |   |       |   |   |   |-- ✅ back.pdf
|   |   |       |   |   |   |-- ✅ back.png
|   |   |       |   |   |   |-- ✅ back.svg
|   |   |       |   |   |   |-- ✅ back_large.png
|   |   |       |   |   |   |-- ✅ filesave-symbolic.svg
|   |   |       |   |   |   |-- ✅ filesave.pdf
|   |   |       |   |   |   |-- ✅ filesave.png
|   |   |       |   |   |   |-- ✅ filesave.svg
|   |   |       |   |   |   |-- ✅ filesave_large.png
|   |   |       |   |   |   |-- ✅ forward-symbolic.svg
|   |   |       |   |   |   |-- ✅ forward.pdf
|   |   |       |   |   |   |-- ✅ forward.png
|   |   |       |   |   |   |-- ✅ forward.svg
|   |   |       |   |   |   |-- ✅ forward_large.png
|   |   |       |   |   |   |-- ✅ hand.pdf
|   |   |       |   |   |   |-- ✅ hand.png
|   |   |       |   |   |   |-- ✅ hand.svg
|   |   |       |   |   |   |-- ✅ help-symbolic.svg
|   |   |       |   |   |   |-- ✅ help.pdf
|   |   |       |   |   |   |-- ✅ help.png
|   |   |       |   |   |   |-- ✅ help.svg
|   |   |       |   |   |   |-- ✅ help_large.png
|   |   |       |   |   |   |-- ✅ home-symbolic.svg
|   |   |       |   |   |   |-- ✅ home.pdf
|   |   |       |   |   |   |-- ✅ home.png
|   |   |       |   |   |   |-- ✅ home.svg
|   |   |       |   |   |   |-- ✅ home_large.png
|   |   |       |   |   |   |-- ✅ matplotlib.pdf
|   |   |       |   |   |   |-- ✅ matplotlib.png
|   |   |       |   |   |   |-- ✅ matplotlib.svg
|   |   |       |   |   |   |-- ✅ matplotlib_large.png
|   |   |       |   |   |   |-- ✅ move-symbolic.svg
|   |   |       |   |   |   |-- ✅ move.pdf
|   |   |       |   |   |   |-- ✅ move.png
|   |   |       |   |   |   |-- ✅ move.svg
|   |   |       |   |   |   |-- ✅ move_large.png
|   |   |       |   |   |   |-- ✅ qt4_editor_options.pdf
|   |   |       |   |   |   |-- ✅ qt4_editor_options.png
|   |   |       |   |   |   |-- ✅ qt4_editor_options.svg
|   |   |       |   |   |   |-- ✅ qt4_editor_options_large.png
|   |   |       |   |   |   |-- ✅ subplots-symbolic.svg
|   |   |       |   |   |   |-- ✅ subplots.pdf
|   |   |       |   |   |   |-- ✅ subplots.png
|   |   |       |   |   |   |-- ✅ subplots.svg
|   |   |       |   |   |   |-- ✅ subplots_large.png
|   |   |       |   |   |   |-- ✅ zoom_to_rect-symbolic.svg
|   |   |       |   |   |   |-- ✅ zoom_to_rect.pdf
|   |   |       |   |   |   |-- ✅ zoom_to_rect.png
|   |   |       |   |   |   |-- ✅ zoom_to_rect.svg
|   |   |       |   |   |   \-- ✅ zoom_to_rect_large.png
|   |   |       |   |   |-- ✅ plot_directive/
|   |   |       |   |   |   \-- ✅ plot_directive.css
|   |   |       |   |   |-- ✅ sample_data/
|   |   |       |   |   |   |-- ✅ axes_grid/
|   |   |       |   |   |   |   \-- ✅ bivariate_normal.npy
|   |   |       |   |   |   |-- ✅ data_x_x2_x3.csv
|   |   |       |   |   |   |-- ✅ eeg.dat
|   |   |       |   |   |   |-- ✅ embedding_in_wx3.xrc
|   |   |       |   |   |   |-- ✅ goog.npz
|   |   |       |   |   |   |-- ✅ grace_hopper.jpg
|   |   |       |   |   |   |-- ✅ jacksboro_fault_dem.npz
|   |   |       |   |   |   |-- ✅ logo2.png
|   |   |       |   |   |   |-- ✅ membrane.dat
|   |   |       |   |   |   |-- ✅ Minduka_Present_Blue_Pack.png
|   |   |       |   |   |   |-- ✅ msft.csv
|   |   |       |   |   |   |-- ✅ README.txt
|   |   |       |   |   |   |-- ✅ s1045.ima.gz
|   |   |       |   |   |   |-- ✅ Stocks.csv
|   |   |       |   |   |   \-- ✅ topobathy.npz
|   |   |       |   |   |-- ✅ stylelib/
|   |   |       |   |   |   |-- ✅ _classic_test_patch.mplstyle
|   |   |       |   |   |   |-- ✅ _mpl-gallery-nogrid.mplstyle
|   |   |       |   |   |   |-- ✅ _mpl-gallery.mplstyle
|   |   |       |   |   |   |-- ✅ bmh.mplstyle
|   |   |       |   |   |   |-- ✅ classic.mplstyle
|   |   |       |   |   |   |-- ✅ dark_background.mplstyle
|   |   |       |   |   |   |-- ✅ fast.mplstyle
|   |   |       |   |   |   |-- ✅ fivethirtyeight.mplstyle
|   |   |       |   |   |   |-- ✅ ggplot.mplstyle
|   |   |       |   |   |   |-- ✅ grayscale.mplstyle
|   |   |       |   |   |   |-- ✅ petroff10.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-bright.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-colorblind.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-dark-palette.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-dark.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-darkgrid.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-deep.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-muted.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-notebook.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-paper.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-pastel.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-poster.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-talk.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-ticks.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-white.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8-whitegrid.mplstyle
|   |   |       |   |   |   |-- ✅ seaborn-v0_8.mplstyle
|   |   |       |   |   |   |-- ✅ Solarize_Light2.mplstyle
|   |   |       |   |   |   \-- ✅ tableau-colorblind10.mplstyle
|   |   |       |   |   |-- ✅ kpsewhich.lua
|   |   |       |   |   \-- ✅ matplotlibrc
|   |   |       |   |-- ✅ projections/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ geo.py
|   |   |       |   |   |-- ✅ geo.pyi
|   |   |       |   |   |-- ✅ polar.py
|   |   |       |   |   \-- ✅ polar.pyi
|   |   |       |   |-- ✅ sphinxext/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ figmpl_directive.py
|   |   |       |   |   |-- ✅ mathmpl.py
|   |   |       |   |   |-- ✅ plot_directive.py
|   |   |       |   |   \-- ✅ roles.py
|   |   |       |   |-- ✅ style/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ core.py
|   |   |       |   |   \-- ✅ core.pyi
|   |   |       |   |-- ✅ testing/
|   |   |       |   |   |-- ✅ jpl_units/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Duration.py
|   |   |       |   |   |   |-- ✅ Epoch.py
|   |   |       |   |   |   |-- ✅ EpochConverter.py
|   |   |       |   |   |   |-- ✅ StrConverter.py
|   |   |       |   |   |   |-- ✅ UnitDbl.py
|   |   |       |   |   |   |-- ✅ UnitDblConverter.py
|   |   |       |   |   |   \-- ✅ UnitDblFormatter.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _markers.py
|   |   |       |   |   |-- ✅ compare.py
|   |   |       |   |   |-- ✅ compare.pyi
|   |   |       |   |   |-- ✅ conftest.py
|   |   |       |   |   |-- ✅ conftest.pyi
|   |   |       |   |   |-- ✅ decorators.py
|   |   |       |   |   |-- ✅ decorators.pyi
|   |   |       |   |   |-- ✅ exceptions.py
|   |   |       |   |   |-- ✅ widgets.py
|   |   |       |   |   \-- ✅ widgets.pyi
|   |   |       |   |-- ✅ tests/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ conftest.py
|   |   |       |   |   |-- ✅ test_afm.py
|   |   |       |   |   |-- ✅ test_agg.py
|   |   |       |   |   |-- ✅ test_agg_filter.py
|   |   |       |   |   |-- ✅ test_animation.py
|   |   |       |   |   |-- ✅ test_api.py
|   |   |       |   |   |-- ✅ test_arrow_patches.py
|   |   |       |   |   |-- ✅ test_artist.py
|   |   |       |   |   |-- ✅ test_axes.py
|   |   |       |   |   |-- ✅ test_axis.py
|   |   |       |   |   |-- ✅ test_backend_bases.py
|   |   |       |   |   |-- ✅ test_backend_cairo.py
|   |   |       |   |   |-- ✅ test_backend_gtk3.py
|   |   |       |   |   |-- ✅ test_backend_inline.py
|   |   |       |   |   |-- ✅ test_backend_macosx.py
|   |   |       |   |   |-- ✅ test_backend_nbagg.py
|   |   |       |   |   |-- ✅ test_backend_pdf.py
|   |   |       |   |   |-- ✅ test_backend_pgf.py
|   |   |       |   |   |-- ✅ test_backend_ps.py
|   |   |       |   |   |-- ✅ test_backend_qt.py
|   |   |       |   |   |-- ✅ test_backend_registry.py
|   |   |       |   |   |-- ✅ test_backend_svg.py
|   |   |       |   |   |-- ✅ test_backend_template.py
|   |   |       |   |   |-- ✅ test_backend_tk.py
|   |   |       |   |   |-- ✅ test_backend_tools.py
|   |   |       |   |   |-- ✅ test_backend_webagg.py
|   |   |       |   |   |-- ✅ test_backends_interactive.py
|   |   |       |   |   |-- ✅ test_basic.py
|   |   |       |   |   |-- ✅ test_bbox_tight.py
|   |   |       |   |   |-- ✅ test_bezier.py
|   |   |       |   |   |-- ✅ test_category.py
|   |   |       |   |   |-- ✅ test_cbook.py
|   |   |       |   |   |-- ✅ test_collections.py
|   |   |       |   |   |-- ✅ test_colorbar.py
|   |   |       |   |   |-- ✅ test_colors.py
|   |   |       |   |   |-- ✅ test_compare_images.py
|   |   |       |   |   |-- ✅ test_constrainedlayout.py
|   |   |       |   |   |-- ✅ test_container.py
|   |   |       |   |   |-- ✅ test_contour.py
|   |   |       |   |   |-- ✅ test_cycles.py
|   |   |       |   |   |-- ✅ test_dates.py
|   |   |       |   |   |-- ✅ test_datetime.py
|   |   |       |   |   |-- ✅ test_determinism.py
|   |   |       |   |   |-- ✅ test_doc.py
|   |   |       |   |   |-- ✅ test_dviread.py
|   |   |       |   |   |-- ✅ test_figure.py
|   |   |       |   |   |-- ✅ test_font_manager.py
|   |   |       |   |   |-- ✅ test_fontconfig_pattern.py
|   |   |       |   |   |-- ✅ test_ft2font.py
|   |   |       |   |   |-- ✅ test_getattr.py
|   |   |       |   |   |-- ✅ test_gridspec.py
|   |   |       |   |   |-- ✅ test_image.py
|   |   |       |   |   |-- ✅ test_legend.py
|   |   |       |   |   |-- ✅ test_lines.py
|   |   |       |   |   |-- ✅ test_marker.py
|   |   |       |   |   |-- ✅ test_mathtext.py
|   |   |       |   |   |-- ✅ test_matplotlib.py
|   |   |       |   |   |-- ✅ test_mlab.py
|   |   |       |   |   |-- ✅ test_multivariate_colormaps.py
|   |   |       |   |   |-- ✅ test_offsetbox.py
|   |   |       |   |   |-- ✅ test_patches.py
|   |   |       |   |   |-- ✅ test_path.py
|   |   |       |   |   |-- ✅ test_patheffects.py
|   |   |       |   |   |-- ✅ test_pickle.py
|   |   |       |   |   |-- ✅ test_png.py
|   |   |       |   |   |-- ✅ test_polar.py
|   |   |       |   |   |-- ✅ test_preprocess_data.py
|   |   |       |   |   |-- ✅ test_pyplot.py
|   |   |       |   |   |-- ✅ test_quiver.py
|   |   |       |   |   |-- ✅ test_rcparams.py
|   |   |       |   |   |-- ✅ test_sankey.py
|   |   |       |   |   |-- ✅ test_scale.py
|   |   |       |   |   |-- ✅ test_simplification.py
|   |   |       |   |   |-- ✅ test_skew.py
|   |   |       |   |   |-- ✅ test_sphinxext.py
|   |   |       |   |   |-- ✅ test_spines.py
|   |   |       |   |   |-- ✅ test_streamplot.py
|   |   |       |   |   |-- ✅ test_style.py
|   |   |       |   |   |-- ✅ test_subplots.py
|   |   |       |   |   |-- ✅ test_table.py
|   |   |       |   |   |-- ✅ test_testing.py
|   |   |       |   |   |-- ✅ test_texmanager.py
|   |   |       |   |   |-- ✅ test_text.py
|   |   |       |   |   |-- ✅ test_textpath.py
|   |   |       |   |   |-- ✅ test_ticker.py
|   |   |       |   |   |-- ✅ test_tightlayout.py
|   |   |       |   |   |-- ✅ test_transforms.py
|   |   |       |   |   |-- ✅ test_triangulation.py
|   |   |       |   |   |-- ✅ test_type1font.py
|   |   |       |   |   |-- ✅ test_units.py
|   |   |       |   |   |-- ✅ test_usetex.py
|   |   |       |   |   \-- ✅ test_widgets.py
|   |   |       |   |-- ✅ tri/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _triangulation.py
|   |   |       |   |   |-- ✅ _triangulation.pyi
|   |   |       |   |   |-- ✅ _tricontour.py
|   |   |       |   |   |-- ✅ _tricontour.pyi
|   |   |       |   |   |-- ✅ _trifinder.py
|   |   |       |   |   |-- ✅ _trifinder.pyi
|   |   |       |   |   |-- ✅ _triinterpolate.py
|   |   |       |   |   |-- ✅ _triinterpolate.pyi
|   |   |       |   |   |-- ✅ _tripcolor.py
|   |   |       |   |   |-- ✅ _tripcolor.pyi
|   |   |       |   |   |-- ✅ _triplot.py
|   |   |       |   |   |-- ✅ _triplot.pyi
|   |   |       |   |   |-- ✅ _trirefine.py
|   |   |       |   |   |-- ✅ _trirefine.pyi
|   |   |       |   |   |-- ✅ _tritools.py
|   |   |       |   |   \-- ✅ _tritools.pyi
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __init__.pyi
|   |   |       |   |-- ✅ _afm.py
|   |   |       |   |-- ✅ _animation_data.py
|   |   |       |   |-- ✅ _blocking_input.py
|   |   |       |   |-- ✅ _c_internal_utils.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _c_internal_utils.pyi
|   |   |       |   |-- ✅ _cm.py
|   |   |       |   |-- ✅ _cm_bivar.py
|   |   |       |   |-- ✅ _cm_listed.py
|   |   |       |   |-- ✅ _cm_multivar.py
|   |   |       |   |-- ✅ _color_data.py
|   |   |       |   |-- ✅ _color_data.pyi
|   |   |       |   |-- ✅ _constrained_layout.py
|   |   |       |   |-- ✅ _docstring.py
|   |   |       |   |-- ✅ _docstring.pyi
|   |   |       |   |-- ✅ _enums.py
|   |   |       |   |-- ✅ _enums.pyi
|   |   |       |   |-- ✅ _fontconfig_pattern.py
|   |   |       |   |-- ✅ _image.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _image.pyi
|   |   |       |   |-- ✅ _internal_utils.py
|   |   |       |   |-- ✅ _layoutgrid.py
|   |   |       |   |-- ✅ _mathtext.py
|   |   |       |   |-- ✅ _mathtext_data.py
|   |   |       |   |-- ✅ _path.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _path.pyi
|   |   |       |   |-- ✅ _pylab_helpers.py
|   |   |       |   |-- ✅ _pylab_helpers.pyi
|   |   |       |   |-- ✅ _qhull.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _qhull.pyi
|   |   |       |   |-- ✅ _text_helpers.py
|   |   |       |   |-- ✅ _tight_bbox.py
|   |   |       |   |-- ✅ _tight_layout.py
|   |   |       |   |-- ✅ _tri.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _tri.pyi
|   |   |       |   |-- ✅ _type1font.py
|   |   |       |   |-- ✅ _version.py
|   |   |       |   |-- ✅ animation.py
|   |   |       |   |-- ✅ animation.pyi
|   |   |       |   |-- ✅ artist.py
|   |   |       |   |-- ✅ artist.pyi
|   |   |       |   |-- ✅ axis.py
|   |   |       |   |-- ✅ axis.pyi
|   |   |       |   |-- ✅ backend_bases.py
|   |   |       |   |-- ✅ backend_bases.pyi
|   |   |       |   |-- ✅ backend_managers.py
|   |   |       |   |-- ✅ backend_managers.pyi
|   |   |       |   |-- ✅ backend_tools.py
|   |   |       |   |-- ✅ backend_tools.pyi
|   |   |       |   |-- ✅ bezier.py
|   |   |       |   |-- ✅ bezier.pyi
|   |   |       |   |-- ✅ category.py
|   |   |       |   |-- ✅ cbook.py
|   |   |       |   |-- ✅ cbook.pyi
|   |   |       |   |-- ✅ cm.py
|   |   |       |   |-- ✅ cm.pyi
|   |   |       |   |-- ✅ collections.py
|   |   |       |   |-- ✅ collections.pyi
|   |   |       |   |-- ✅ colorbar.py
|   |   |       |   |-- ✅ colorbar.pyi
|   |   |       |   |-- ✅ colorizer.py
|   |   |       |   |-- ✅ colorizer.pyi
|   |   |       |   |-- ✅ colors.py
|   |   |       |   |-- ✅ colors.pyi
|   |   |       |   |-- ✅ container.py
|   |   |       |   |-- ✅ container.pyi
|   |   |       |   |-- ✅ contour.py
|   |   |       |   |-- ✅ contour.pyi
|   |   |       |   |-- ✅ dates.py
|   |   |       |   |-- ✅ dviread.py
|   |   |       |   |-- ✅ dviread.pyi
|   |   |       |   |-- ✅ figure.py
|   |   |       |   |-- ✅ figure.pyi
|   |   |       |   |-- ✅ font_manager.py
|   |   |       |   |-- ✅ font_manager.pyi
|   |   |       |   |-- ✅ ft2font.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ ft2font.pyi
|   |   |       |   |-- ✅ gridspec.py
|   |   |       |   |-- ✅ gridspec.pyi
|   |   |       |   |-- ✅ hatch.py
|   |   |       |   |-- ✅ hatch.pyi
|   |   |       |   |-- ✅ image.py
|   |   |       |   |-- ✅ image.pyi
|   |   |       |   |-- ✅ inset.py
|   |   |       |   |-- ✅ inset.pyi
|   |   |       |   |-- ✅ layout_engine.py
|   |   |       |   |-- ✅ layout_engine.pyi
|   |   |       |   |-- ✅ legend.py
|   |   |       |   |-- ✅ legend.pyi
|   |   |       |   |-- ✅ legend_handler.py
|   |   |       |   |-- ✅ legend_handler.pyi
|   |   |       |   |-- ✅ lines.py
|   |   |       |   |-- ✅ lines.pyi
|   |   |       |   |-- ✅ markers.py
|   |   |       |   |-- ✅ markers.pyi
|   |   |       |   |-- ✅ mathtext.py
|   |   |       |   |-- ✅ mathtext.pyi
|   |   |       |   |-- ✅ mlab.py
|   |   |       |   |-- ✅ mlab.pyi
|   |   |       |   |-- ✅ offsetbox.py
|   |   |       |   |-- ✅ offsetbox.pyi
|   |   |       |   |-- ✅ patches.py
|   |   |       |   |-- ✅ patches.pyi
|   |   |       |   |-- ✅ path.py
|   |   |       |   |-- ✅ path.pyi
|   |   |       |   |-- ✅ patheffects.py
|   |   |       |   |-- ✅ patheffects.pyi
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ pylab.py
|   |   |       |   |-- ✅ pyplot.py
|   |   |       |   |-- ✅ quiver.py
|   |   |       |   |-- ✅ quiver.pyi
|   |   |       |   |-- ✅ rcsetup.py
|   |   |       |   |-- ✅ rcsetup.pyi
|   |   |       |   |-- ✅ sankey.py
|   |   |       |   |-- ✅ sankey.pyi
|   |   |       |   |-- ✅ scale.py
|   |   |       |   |-- ✅ scale.pyi
|   |   |       |   |-- ✅ spines.py
|   |   |       |   |-- ✅ spines.pyi
|   |   |       |   |-- ✅ stackplot.py
|   |   |       |   |-- ✅ stackplot.pyi
|   |   |       |   |-- ✅ streamplot.py
|   |   |       |   |-- ✅ streamplot.pyi
|   |   |       |   |-- ✅ table.py
|   |   |       |   |-- ✅ table.pyi
|   |   |       |   |-- ✅ texmanager.py
|   |   |       |   |-- ✅ texmanager.pyi
|   |   |       |   |-- ✅ text.py
|   |   |       |   |-- ✅ text.pyi
|   |   |       |   |-- ✅ textpath.py
|   |   |       |   |-- ✅ textpath.pyi
|   |   |       |   |-- ✅ ticker.py
|   |   |       |   |-- ✅ ticker.pyi
|   |   |       |   |-- ✅ transforms.py
|   |   |       |   |-- ✅ transforms.pyi
|   |   |       |   |-- ✅ typing.py
|   |   |       |   |-- ✅ units.py
|   |   |       |   |-- ✅ widgets.py
|   |   |       |   \-- ✅ widgets.pyi
|   |   |       |-- ✅ matplotlib-3.10.6.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ mpl_toolkits/
|   |   |       |   |-- ✅ axes_grid1/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   \-- ✅ test_axes_grid1.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ anchored_artists.py
|   |   |       |   |   |-- ✅ axes_divider.py
|   |   |       |   |   |-- ✅ axes_grid.py
|   |   |       |   |   |-- ✅ axes_rgb.py
|   |   |       |   |   |-- ✅ axes_size.py
|   |   |       |   |   |-- ✅ inset_locator.py
|   |   |       |   |   |-- ✅ mpl_axes.py
|   |   |       |   |   \-- ✅ parasite_axes.py
|   |   |       |   |-- ✅ axisartist/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_angle_helper.py
|   |   |       |   |   |   |-- ✅ test_axis_artist.py
|   |   |       |   |   |   |-- ✅ test_axislines.py
|   |   |       |   |   |   |-- ✅ test_floating_axes.py
|   |   |       |   |   |   |-- ✅ test_grid_finder.py
|   |   |       |   |   |   \-- ✅ test_grid_helper_curvelinear.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ angle_helper.py
|   |   |       |   |   |-- ✅ axes_divider.py
|   |   |       |   |   |-- ✅ axis_artist.py
|   |   |       |   |   |-- ✅ axisline_style.py
|   |   |       |   |   |-- ✅ axislines.py
|   |   |       |   |   |-- ✅ floating_axes.py
|   |   |       |   |   |-- ✅ grid_finder.py
|   |   |       |   |   |-- ✅ grid_helper_curvelinear.py
|   |   |       |   |   \-- ✅ parasite_axes.py
|   |   |       |   \-- ✅ mplot3d/
|   |   |       |       |-- ✅ tests/
|   |   |       |       |   |-- ✅ __init__.py
|   |   |       |       |   |-- ✅ conftest.py
|   |   |       |       |   |-- ✅ test_art3d.py
|   |   |       |       |   |-- ✅ test_axes3d.py
|   |   |       |       |   \-- ✅ test_legend3d.py
|   |   |       |       |-- ✅ __init__.py
|   |   |       |       |-- ✅ art3d.py
|   |   |       |       |-- ✅ axes3d.py
|   |   |       |       |-- ✅ axis3d.py
|   |   |       |       \-- ✅ proj3d.py
|   |   |       |-- ✅ multidict/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _abc.py
|   |   |       |   |-- ✅ _compat.py
|   |   |       |   |-- ✅ _multidict.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _multidict_py.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ multidict-6.6.4.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ numpy/
|   |   |       |   |-- ✅ _core/
|   |   |       |   |   |-- ✅ include/
|   |   |       |   |   |   \-- ✅ numpy/
|   |   |       |   |   |       |-- ✅ random/
|   |   |       |   |   |       |   |-- ✅ bitgen.h
|   |   |       |   |   |       |   |-- ✅ distributions.h
|   |   |       |   |   |       |   |-- ✅ libdivide.h
|   |   |       |   |   |       |   \-- ✅ LICENSE.txt
|   |   |       |   |   |       |-- ✅ __multiarray_api.c
|   |   |       |   |   |       |-- ✅ __multiarray_api.h
|   |   |       |   |   |       |-- ✅ __ufunc_api.c
|   |   |       |   |   |       |-- ✅ __ufunc_api.h
|   |   |       |   |   |       |-- ✅ _neighborhood_iterator_imp.h
|   |   |       |   |   |       |-- ✅ _numpyconfig.h
|   |   |       |   |   |       |-- ✅ _public_dtype_api_table.h
|   |   |       |   |   |       |-- ✅ arrayobject.h
|   |   |       |   |   |       |-- ✅ arrayscalars.h
|   |   |       |   |   |       |-- ✅ dtype_api.h
|   |   |       |   |   |       |-- ✅ halffloat.h
|   |   |       |   |   |       |-- ✅ ndarrayobject.h
|   |   |       |   |   |       |-- ✅ ndarraytypes.h
|   |   |       |   |   |       |-- ✅ npy_2_compat.h
|   |   |       |   |   |       |-- ✅ npy_2_complexcompat.h
|   |   |       |   |   |       |-- ✅ npy_3kcompat.h
|   |   |       |   |   |       |-- ✅ npy_common.h
|   |   |       |   |   |       |-- ✅ npy_cpu.h
|   |   |       |   |   |       |-- ✅ npy_endian.h
|   |   |       |   |   |       |-- ✅ npy_math.h
|   |   |       |   |   |       |-- ✅ npy_no_deprecated_api.h
|   |   |       |   |   |       |-- ✅ npy_os.h
|   |   |       |   |   |       |-- ✅ numpyconfig.h
|   |   |       |   |   |       |-- ✅ ufuncobject.h
|   |   |       |   |   |       \-- ✅ utils.h
|   |   |       |   |   |-- ✅ lib/
|   |   |       |   |   |   |-- ✅ npy-pkg-config/
|   |   |       |   |   |   |   |-- ✅ mlib.ini
|   |   |       |   |   |   |   \-- ✅ npymath.ini
|   |   |       |   |   |   |-- ✅ pkgconfig/
|   |   |       |   |   |   |   \-- ✅ numpy.pc
|   |   |       |   |   |   \-- ✅ npymath.lib
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ astype_copy.pkl
|   |   |       |   |   |   |   |-- ✅ generate_umath_validation_data.cpp
|   |   |       |   |   |   |   |-- ✅ recarray_from_file.fits
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-arccos.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-arccosh.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-arcsin.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-arcsinh.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-arctan.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-arctanh.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-cbrt.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-cos.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-cosh.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-exp.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-exp2.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-expm1.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-log.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-log10.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-log1p.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-log2.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-README.txt
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-sin.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-sinh.csv
|   |   |       |   |   |   |   |-- ✅ umath-validation-set-tan.csv
|   |   |       |   |   |   |   \-- ✅ umath-validation-set-tanh.csv
|   |   |       |   |   |   |-- ✅ examples/
|   |   |       |   |   |   |   |-- ✅ cython/
|   |   |       |   |   |   |   |   |-- ✅ checks.pyx
|   |   |       |   |   |   |   |   |-- ✅ meson.build
|   |   |       |   |   |   |   |   \-- ✅ setup.py
|   |   |       |   |   |   |   \-- ✅ limited_api/
|   |   |       |   |   |   |       |-- ✅ limited_api1.c
|   |   |       |   |   |   |       |-- ✅ limited_api2.pyx
|   |   |       |   |   |   |       |-- ✅ limited_api_latest.c
|   |   |       |   |   |   |       |-- ✅ meson.build
|   |   |       |   |   |   |       \-- ✅ setup.py
|   |   |       |   |   |   |-- ✅ _locales.py
|   |   |       |   |   |   |-- ✅ _natype.py
|   |   |       |   |   |   |-- ✅ test__exceptions.py
|   |   |       |   |   |   |-- ✅ test_abc.py
|   |   |       |   |   |   |-- ✅ test_api.py
|   |   |       |   |   |   |-- ✅ test_argparse.py
|   |   |       |   |   |   |-- ✅ test_array_api_info.py
|   |   |       |   |   |   |-- ✅ test_array_coercion.py
|   |   |       |   |   |   |-- ✅ test_array_interface.py
|   |   |       |   |   |   |-- ✅ test_arraymethod.py
|   |   |       |   |   |   |-- ✅ test_arrayobject.py
|   |   |       |   |   |   |-- ✅ test_arrayprint.py
|   |   |       |   |   |   |-- ✅ test_casting_floatingpoint_errors.py
|   |   |       |   |   |   |-- ✅ test_casting_unittests.py
|   |   |       |   |   |   |-- ✅ test_conversion_utils.py
|   |   |       |   |   |   |-- ✅ test_cpu_dispatcher.py
|   |   |       |   |   |   |-- ✅ test_cpu_features.py
|   |   |       |   |   |   |-- ✅ test_custom_dtypes.py
|   |   |       |   |   |   |-- ✅ test_cython.py
|   |   |       |   |   |   |-- ✅ test_datetime.py
|   |   |       |   |   |   |-- ✅ test_defchararray.py
|   |   |       |   |   |   |-- ✅ test_deprecations.py
|   |   |       |   |   |   |-- ✅ test_dlpack.py
|   |   |       |   |   |   |-- ✅ test_dtype.py
|   |   |       |   |   |   |-- ✅ test_einsum.py
|   |   |       |   |   |   |-- ✅ test_errstate.py
|   |   |       |   |   |   |-- ✅ test_extint128.py
|   |   |       |   |   |   |-- ✅ test_function_base.py
|   |   |       |   |   |   |-- ✅ test_getlimits.py
|   |   |       |   |   |   |-- ✅ test_half.py
|   |   |       |   |   |   |-- ✅ test_hashtable.py
|   |   |       |   |   |   |-- ✅ test_indexerrors.py
|   |   |       |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |-- ✅ test_item_selection.py
|   |   |       |   |   |   |-- ✅ test_limited_api.py
|   |   |       |   |   |   |-- ✅ test_longdouble.py
|   |   |       |   |   |   |-- ✅ test_machar.py
|   |   |       |   |   |   |-- ✅ test_mem_overlap.py
|   |   |       |   |   |   |-- ✅ test_mem_policy.py
|   |   |       |   |   |   |-- ✅ test_memmap.py
|   |   |       |   |   |   |-- ✅ test_multiarray.py
|   |   |       |   |   |   |-- ✅ test_multithreading.py
|   |   |       |   |   |   |-- ✅ test_nditer.py
|   |   |       |   |   |   |-- ✅ test_nep50_promotions.py
|   |   |       |   |   |   |-- ✅ test_numeric.py
|   |   |       |   |   |   |-- ✅ test_numerictypes.py
|   |   |       |   |   |   |-- ✅ test_overrides.py
|   |   |       |   |   |   |-- ✅ test_print.py
|   |   |       |   |   |   |-- ✅ test_protocols.py
|   |   |       |   |   |   |-- ✅ test_records.py
|   |   |       |   |   |   |-- ✅ test_regression.py
|   |   |       |   |   |   |-- ✅ test_scalar_ctors.py
|   |   |       |   |   |   |-- ✅ test_scalar_methods.py
|   |   |       |   |   |   |-- ✅ test_scalarbuffer.py
|   |   |       |   |   |   |-- ✅ test_scalarinherit.py
|   |   |       |   |   |   |-- ✅ test_scalarmath.py
|   |   |       |   |   |   |-- ✅ test_scalarprint.py
|   |   |       |   |   |   |-- ✅ test_shape_base.py
|   |   |       |   |   |   |-- ✅ test_simd.py
|   |   |       |   |   |   |-- ✅ test_simd_module.py
|   |   |       |   |   |   |-- ✅ test_stringdtype.py
|   |   |       |   |   |   |-- ✅ test_strings.py
|   |   |       |   |   |   |-- ✅ test_ufunc.py
|   |   |       |   |   |   |-- ✅ test_umath.py
|   |   |       |   |   |   |-- ✅ test_umath_accuracy.py
|   |   |       |   |   |   |-- ✅ test_umath_complex.py
|   |   |       |   |   |   \-- ✅ test_unicode.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _add_newdocs.py
|   |   |       |   |   |-- ✅ _add_newdocs.pyi
|   |   |       |   |   |-- ✅ _add_newdocs_scalars.py
|   |   |       |   |   |-- ✅ _add_newdocs_scalars.pyi
|   |   |       |   |   |-- ✅ _asarray.py
|   |   |       |   |   |-- ✅ _asarray.pyi
|   |   |       |   |   |-- ✅ _dtype.py
|   |   |       |   |   |-- ✅ _dtype.pyi
|   |   |       |   |   |-- ✅ _dtype_ctypes.py
|   |   |       |   |   |-- ✅ _dtype_ctypes.pyi
|   |   |       |   |   |-- ✅ _exceptions.py
|   |   |       |   |   |-- ✅ _exceptions.pyi
|   |   |       |   |   |-- ✅ _internal.py
|   |   |       |   |   |-- ✅ _internal.pyi
|   |   |       |   |   |-- ✅ _machar.py
|   |   |       |   |   |-- ✅ _machar.pyi
|   |   |       |   |   |-- ✅ _methods.py
|   |   |       |   |   |-- ✅ _methods.pyi
|   |   |       |   |   |-- ✅ _multiarray_tests.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _multiarray_tests.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _multiarray_umath.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _multiarray_umath.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _operand_flag_tests.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _operand_flag_tests.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _rational_tests.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _rational_tests.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _simd.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _simd.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _simd.pyi
|   |   |       |   |   |-- ✅ _string_helpers.py
|   |   |       |   |   |-- ✅ _string_helpers.pyi
|   |   |       |   |   |-- ✅ _struct_ufunc_tests.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _struct_ufunc_tests.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _type_aliases.py
|   |   |       |   |   |-- ✅ _type_aliases.pyi
|   |   |       |   |   |-- ✅ _ufunc_config.py
|   |   |       |   |   |-- ✅ _ufunc_config.pyi
|   |   |       |   |   |-- ✅ _umath_tests.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _umath_tests.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ arrayprint.py
|   |   |       |   |   |-- ✅ arrayprint.pyi
|   |   |       |   |   |-- ✅ cversions.py
|   |   |       |   |   |-- ✅ defchararray.py
|   |   |       |   |   |-- ✅ defchararray.pyi
|   |   |       |   |   |-- ✅ einsumfunc.py
|   |   |       |   |   |-- ✅ einsumfunc.pyi
|   |   |       |   |   |-- ✅ fromnumeric.py
|   |   |       |   |   |-- ✅ fromnumeric.pyi
|   |   |       |   |   |-- ✅ function_base.py
|   |   |       |   |   |-- ✅ function_base.pyi
|   |   |       |   |   |-- ✅ getlimits.py
|   |   |       |   |   |-- ✅ getlimits.pyi
|   |   |       |   |   |-- ✅ memmap.py
|   |   |       |   |   |-- ✅ memmap.pyi
|   |   |       |   |   |-- ✅ multiarray.py
|   |   |       |   |   |-- ✅ multiarray.pyi
|   |   |       |   |   |-- ✅ numeric.py
|   |   |       |   |   |-- ✅ numeric.pyi
|   |   |       |   |   |-- ✅ numerictypes.py
|   |   |       |   |   |-- ✅ numerictypes.pyi
|   |   |       |   |   |-- ✅ overrides.py
|   |   |       |   |   |-- ✅ overrides.pyi
|   |   |       |   |   |-- ✅ printoptions.py
|   |   |       |   |   |-- ✅ printoptions.pyi
|   |   |       |   |   |-- ✅ records.py
|   |   |       |   |   |-- ✅ records.pyi
|   |   |       |   |   |-- ✅ shape_base.py
|   |   |       |   |   |-- ✅ shape_base.pyi
|   |   |       |   |   |-- ✅ strings.py
|   |   |       |   |   |-- ✅ strings.pyi
|   |   |       |   |   |-- ✅ umath.py
|   |   |       |   |   \-- ✅ umath.pyi
|   |   |       |   |-- ✅ _pyinstaller/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ pyinstaller-smoke.py
|   |   |       |   |   |   \-- ✅ test_pyinstaller.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ hook-numpy.py
|   |   |       |   |   \-- ✅ hook-numpy.pyi
|   |   |       |   |-- ✅ _typing/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _add_docstring.py
|   |   |       |   |   |-- ✅ _array_like.py
|   |   |       |   |   |-- ✅ _callable.pyi
|   |   |       |   |   |-- ✅ _char_codes.py
|   |   |       |   |   |-- ✅ _dtype_like.py
|   |   |       |   |   |-- ✅ _extended_precision.py
|   |   |       |   |   |-- ✅ _nbit.py
|   |   |       |   |   |-- ✅ _nbit_base.py
|   |   |       |   |   |-- ✅ _nbit_base.pyi
|   |   |       |   |   |-- ✅ _nested_sequence.py
|   |   |       |   |   |-- ✅ _scalars.py
|   |   |       |   |   |-- ✅ _shape.py
|   |   |       |   |   |-- ✅ _ufunc.py
|   |   |       |   |   \-- ✅ _ufunc.pyi
|   |   |       |   |-- ✅ _utils/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _convertions.py
|   |   |       |   |   |-- ✅ _convertions.pyi
|   |   |       |   |   |-- ✅ _inspect.py
|   |   |       |   |   |-- ✅ _inspect.pyi
|   |   |       |   |   |-- ✅ _pep440.py
|   |   |       |   |   \-- ✅ _pep440.pyi
|   |   |       |   |-- ✅ char/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ __init__.pyi
|   |   |       |   |-- ✅ core/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _dtype.py
|   |   |       |   |   |-- ✅ _dtype.pyi
|   |   |       |   |   |-- ✅ _dtype_ctypes.py
|   |   |       |   |   |-- ✅ _dtype_ctypes.pyi
|   |   |       |   |   |-- ✅ _internal.py
|   |   |       |   |   |-- ✅ _multiarray_umath.py
|   |   |       |   |   |-- ✅ _utils.py
|   |   |       |   |   |-- ✅ arrayprint.py
|   |   |       |   |   |-- ✅ defchararray.py
|   |   |       |   |   |-- ✅ einsumfunc.py
|   |   |       |   |   |-- ✅ fromnumeric.py
|   |   |       |   |   |-- ✅ function_base.py
|   |   |       |   |   |-- ✅ getlimits.py
|   |   |       |   |   |-- ✅ multiarray.py
|   |   |       |   |   |-- ✅ numeric.py
|   |   |       |   |   |-- ✅ numerictypes.py
|   |   |       |   |   |-- ✅ overrides.py
|   |   |       |   |   |-- ✅ overrides.pyi
|   |   |       |   |   |-- ✅ records.py
|   |   |       |   |   |-- ✅ shape_base.py
|   |   |       |   |   \-- ✅ umath.py
|   |   |       |   |-- ✅ ctypeslib/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _ctypeslib.py
|   |   |       |   |   \-- ✅ _ctypeslib.pyi
|   |   |       |   |-- ✅ doc/
|   |   |       |   |   \-- ✅ ufuncs.py
|   |   |       |   |-- ✅ f2py/
|   |   |       |   |   |-- ✅ _backends/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |   |-- ✅ _backend.py
|   |   |       |   |   |   |-- ✅ _backend.pyi
|   |   |       |   |   |   |-- ✅ _distutils.py
|   |   |       |   |   |   |-- ✅ _distutils.pyi
|   |   |       |   |   |   |-- ✅ _meson.py
|   |   |       |   |   |   |-- ✅ _meson.pyi
|   |   |       |   |   |   \-- ✅ meson.build.template
|   |   |       |   |   |-- ✅ src/
|   |   |       |   |   |   |-- ✅ fortranobject.c
|   |   |       |   |   |   \-- ✅ fortranobject.h
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ src/
|   |   |       |   |   |   |   |-- ✅ abstract_interface/
|   |   |       |   |   |   |   |   |-- ✅ foo.f90
|   |   |       |   |   |   |   |   \-- ✅ gh18403_mod.f90
|   |   |       |   |   |   |   |-- ✅ array_from_pyobj/
|   |   |       |   |   |   |   |   \-- ✅ wrapmodule.c
|   |   |       |   |   |   |   |-- ✅ assumed_shape/
|   |   |       |   |   |   |   |   |-- ✅ .f2py_f2cmap
|   |   |       |   |   |   |   |   |-- ✅ foo_free.f90
|   |   |       |   |   |   |   |   |-- ✅ foo_mod.f90
|   |   |       |   |   |   |   |   |-- ✅ foo_use.f90
|   |   |       |   |   |   |   |   \-- ✅ precision.f90
|   |   |       |   |   |   |   |-- ✅ block_docstring/
|   |   |       |   |   |   |   |   \-- ✅ foo.f
|   |   |       |   |   |   |   |-- ✅ callback/
|   |   |       |   |   |   |   |   |-- ✅ foo.f
|   |   |       |   |   |   |   |   |-- ✅ gh17797.f90
|   |   |       |   |   |   |   |   |-- ✅ gh18335.f90
|   |   |       |   |   |   |   |   |-- ✅ gh25211.f
|   |   |       |   |   |   |   |   |-- ✅ gh25211.pyf
|   |   |       |   |   |   |   |   \-- ✅ gh26681.f90
|   |   |       |   |   |   |   |-- ✅ cli/
|   |   |       |   |   |   |   |   |-- ✅ gh_22819.pyf
|   |   |       |   |   |   |   |   |-- ✅ hi77.f
|   |   |       |   |   |   |   |   \-- ✅ hiworld.f90
|   |   |       |   |   |   |   |-- ✅ common/
|   |   |       |   |   |   |   |   |-- ✅ block.f
|   |   |       |   |   |   |   |   \-- ✅ gh19161.f90
|   |   |       |   |   |   |   |-- ✅ crackfortran/
|   |   |       |   |   |   |   |   |-- ✅ accesstype.f90
|   |   |       |   |   |   |   |   |-- ✅ common_with_division.f
|   |   |       |   |   |   |   |   |-- ✅ data_common.f
|   |   |       |   |   |   |   |   |-- ✅ data_multiplier.f
|   |   |       |   |   |   |   |   |-- ✅ data_stmts.f90
|   |   |       |   |   |   |   |   |-- ✅ data_with_comments.f
|   |   |       |   |   |   |   |   |-- ✅ foo_deps.f90
|   |   |       |   |   |   |   |   |-- ✅ gh15035.f
|   |   |       |   |   |   |   |   |-- ✅ gh17859.f
|   |   |       |   |   |   |   |   |-- ✅ gh22648.pyf
|   |   |       |   |   |   |   |   |-- ✅ gh23533.f
|   |   |       |   |   |   |   |   |-- ✅ gh23598.f90
|   |   |       |   |   |   |   |   |-- ✅ gh23598Warn.f90
|   |   |       |   |   |   |   |   |-- ✅ gh23879.f90
|   |   |       |   |   |   |   |   |-- ✅ gh27697.f90
|   |   |       |   |   |   |   |   |-- ✅ gh2848.f90
|   |   |       |   |   |   |   |   |-- ✅ operators.f90
|   |   |       |   |   |   |   |   |-- ✅ privatemod.f90
|   |   |       |   |   |   |   |   |-- ✅ publicmod.f90
|   |   |       |   |   |   |   |   |-- ✅ pubprivmod.f90
|   |   |       |   |   |   |   |   \-- ✅ unicode_comment.f90
|   |   |       |   |   |   |   |-- ✅ f2cmap/
|   |   |       |   |   |   |   |   |-- ✅ .f2py_f2cmap
|   |   |       |   |   |   |   |   \-- ✅ isoFortranEnvMap.f90
|   |   |       |   |   |   |   |-- ✅ isocintrin/
|   |   |       |   |   |   |   |   \-- ✅ isoCtests.f90
|   |   |       |   |   |   |   |-- ✅ kind/
|   |   |       |   |   |   |   |   \-- ✅ foo.f90
|   |   |       |   |   |   |   |-- ✅ mixed/
|   |   |       |   |   |   |   |   |-- ✅ foo.f
|   |   |       |   |   |   |   |   |-- ✅ foo_fixed.f90
|   |   |       |   |   |   |   |   \-- ✅ foo_free.f90
|   |   |       |   |   |   |   |-- ✅ modules/
|   |   |       |   |   |   |   |   |-- ✅ gh25337/
|   |   |       |   |   |   |   |   |   |-- ✅ data.f90
|   |   |       |   |   |   |   |   |   \-- ✅ use_data.f90
|   |   |       |   |   |   |   |   |-- ✅ gh26920/
|   |   |       |   |   |   |   |   |   |-- ✅ two_mods_with_no_public_entities.f90
|   |   |       |   |   |   |   |   |   \-- ✅ two_mods_with_one_public_routine.f90
|   |   |       |   |   |   |   |   |-- ✅ module_data_docstring.f90
|   |   |       |   |   |   |   |   \-- ✅ use_modules.f90
|   |   |       |   |   |   |   |-- ✅ negative_bounds/
|   |   |       |   |   |   |   |   \-- ✅ issue_20853.f90
|   |   |       |   |   |   |   |-- ✅ parameter/
|   |   |       |   |   |   |   |   |-- ✅ constant_array.f90
|   |   |       |   |   |   |   |   |-- ✅ constant_both.f90
|   |   |       |   |   |   |   |   |-- ✅ constant_compound.f90
|   |   |       |   |   |   |   |   |-- ✅ constant_integer.f90
|   |   |       |   |   |   |   |   |-- ✅ constant_non_compound.f90
|   |   |       |   |   |   |   |   \-- ✅ constant_real.f90
|   |   |       |   |   |   |   |-- ✅ quoted_character/
|   |   |       |   |   |   |   |   \-- ✅ foo.f
|   |   |       |   |   |   |   |-- ✅ regression/
|   |   |       |   |   |   |   |   |-- ✅ AB.inc
|   |   |       |   |   |   |   |   |-- ✅ assignOnlyModule.f90
|   |   |       |   |   |   |   |   |-- ✅ datonly.f90
|   |   |       |   |   |   |   |   |-- ✅ f77comments.f
|   |   |       |   |   |   |   |   |-- ✅ f77fixedform.f95
|   |   |       |   |   |   |   |   |-- ✅ f90continuation.f90
|   |   |       |   |   |   |   |   |-- ✅ incfile.f90
|   |   |       |   |   |   |   |   |-- ✅ inout.f90
|   |   |       |   |   |   |   |   |-- ✅ lower_f2py_fortran.f90
|   |   |       |   |   |   |   |   \-- ✅ mod_derived_types.f90
|   |   |       |   |   |   |   |-- ✅ return_character/
|   |   |       |   |   |   |   |   |-- ✅ foo77.f
|   |   |       |   |   |   |   |   \-- ✅ foo90.f90
|   |   |       |   |   |   |   |-- ✅ return_complex/
|   |   |       |   |   |   |   |   |-- ✅ foo77.f
|   |   |       |   |   |   |   |   \-- ✅ foo90.f90
|   |   |       |   |   |   |   |-- ✅ return_integer/
|   |   |       |   |   |   |   |   |-- ✅ foo77.f
|   |   |       |   |   |   |   |   \-- ✅ foo90.f90
|   |   |       |   |   |   |   |-- ✅ return_logical/
|   |   |       |   |   |   |   |   |-- ✅ foo77.f
|   |   |       |   |   |   |   |   \-- ✅ foo90.f90
|   |   |       |   |   |   |   |-- ✅ return_real/
|   |   |       |   |   |   |   |   |-- ✅ foo77.f
|   |   |       |   |   |   |   |   \-- ✅ foo90.f90
|   |   |       |   |   |   |   |-- ✅ routines/
|   |   |       |   |   |   |   |   |-- ✅ funcfortranname.f
|   |   |       |   |   |   |   |   |-- ✅ funcfortranname.pyf
|   |   |       |   |   |   |   |   |-- ✅ subrout.f
|   |   |       |   |   |   |   |   \-- ✅ subrout.pyf
|   |   |       |   |   |   |   |-- ✅ size/
|   |   |       |   |   |   |   |   \-- ✅ foo.f90
|   |   |       |   |   |   |   |-- ✅ string/
|   |   |       |   |   |   |   |   |-- ✅ char.f90
|   |   |       |   |   |   |   |   |-- ✅ fixed_string.f90
|   |   |       |   |   |   |   |   |-- ✅ gh24008.f
|   |   |       |   |   |   |   |   |-- ✅ gh24662.f90
|   |   |       |   |   |   |   |   |-- ✅ gh25286.f90
|   |   |       |   |   |   |   |   |-- ✅ gh25286.pyf
|   |   |       |   |   |   |   |   |-- ✅ gh25286_bc.pyf
|   |   |       |   |   |   |   |   |-- ✅ scalar_string.f90
|   |   |       |   |   |   |   |   \-- ✅ string.f
|   |   |       |   |   |   |   \-- ✅ value_attrspec/
|   |   |       |   |   |   |       \-- ✅ gh21665.f90
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_abstract_interface.py
|   |   |       |   |   |   |-- ✅ test_array_from_pyobj.py
|   |   |       |   |   |   |-- ✅ test_assumed_shape.py
|   |   |       |   |   |   |-- ✅ test_block_docstring.py
|   |   |       |   |   |   |-- ✅ test_callback.py
|   |   |       |   |   |   |-- ✅ test_character.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_crackfortran.py
|   |   |       |   |   |   |-- ✅ test_data.py
|   |   |       |   |   |   |-- ✅ test_docs.py
|   |   |       |   |   |   |-- ✅ test_f2cmap.py
|   |   |       |   |   |   |-- ✅ test_f2py2e.py
|   |   |       |   |   |   |-- ✅ test_isoc.py
|   |   |       |   |   |   |-- ✅ test_kind.py
|   |   |       |   |   |   |-- ✅ test_mixed.py
|   |   |       |   |   |   |-- ✅ test_modules.py
|   |   |       |   |   |   |-- ✅ test_parameter.py
|   |   |       |   |   |   |-- ✅ test_pyf_src.py
|   |   |       |   |   |   |-- ✅ test_quoted_character.py
|   |   |       |   |   |   |-- ✅ test_regression.py
|   |   |       |   |   |   |-- ✅ test_return_character.py
|   |   |       |   |   |   |-- ✅ test_return_complex.py
|   |   |       |   |   |   |-- ✅ test_return_integer.py
|   |   |       |   |   |   |-- ✅ test_return_logical.py
|   |   |       |   |   |   |-- ✅ test_return_real.py
|   |   |       |   |   |   |-- ✅ test_routines.py
|   |   |       |   |   |   |-- ✅ test_semicolon_split.py
|   |   |       |   |   |   |-- ✅ test_size.py
|   |   |       |   |   |   |-- ✅ test_string.py
|   |   |       |   |   |   |-- ✅ test_symbolic.py
|   |   |       |   |   |   |-- ✅ test_value_attrspec.py
|   |   |       |   |   |   \-- ✅ util.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ __main__.py
|   |   |       |   |   |-- ✅ __version__.py
|   |   |       |   |   |-- ✅ __version__.pyi
|   |   |       |   |   |-- ✅ _isocbind.py
|   |   |       |   |   |-- ✅ _isocbind.pyi
|   |   |       |   |   |-- ✅ _src_pyf.py
|   |   |       |   |   |-- ✅ _src_pyf.pyi
|   |   |       |   |   |-- ✅ auxfuncs.py
|   |   |       |   |   |-- ✅ auxfuncs.pyi
|   |   |       |   |   |-- ✅ capi_maps.py
|   |   |       |   |   |-- ✅ capi_maps.pyi
|   |   |       |   |   |-- ✅ cb_rules.py
|   |   |       |   |   |-- ✅ cb_rules.pyi
|   |   |       |   |   |-- ✅ cfuncs.py
|   |   |       |   |   |-- ✅ cfuncs.pyi
|   |   |       |   |   |-- ✅ common_rules.py
|   |   |       |   |   |-- ✅ common_rules.pyi
|   |   |       |   |   |-- ✅ crackfortran.py
|   |   |       |   |   |-- ✅ crackfortran.pyi
|   |   |       |   |   |-- ✅ diagnose.py
|   |   |       |   |   |-- ✅ diagnose.pyi
|   |   |       |   |   |-- ✅ f2py2e.py
|   |   |       |   |   |-- ✅ f2py2e.pyi
|   |   |       |   |   |-- ✅ f90mod_rules.py
|   |   |       |   |   |-- ✅ f90mod_rules.pyi
|   |   |       |   |   |-- ✅ func2subr.py
|   |   |       |   |   |-- ✅ func2subr.pyi
|   |   |       |   |   |-- ✅ rules.py
|   |   |       |   |   |-- ✅ rules.pyi
|   |   |       |   |   |-- ✅ setup.cfg
|   |   |       |   |   |-- ✅ symbolic.py
|   |   |       |   |   |-- ✅ symbolic.pyi
|   |   |       |   |   |-- ✅ use_rules.py
|   |   |       |   |   \-- ✅ use_rules.pyi
|   |   |       |   |-- ✅ fft/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_helper.py
|   |   |       |   |   |   \-- ✅ test_pocketfft.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _helper.py
|   |   |       |   |   |-- ✅ _helper.pyi
|   |   |       |   |   |-- ✅ _pocketfft.py
|   |   |       |   |   |-- ✅ _pocketfft.pyi
|   |   |       |   |   |-- ✅ _pocketfft_umath.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _pocketfft_umath.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ helper.py
|   |   |       |   |   \-- ✅ helper.pyi
|   |   |       |   |-- ✅ lib/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ py2-np0-objarr.npy
|   |   |       |   |   |   |   |-- ✅ py2-objarr.npy
|   |   |       |   |   |   |   |-- ✅ py2-objarr.npz
|   |   |       |   |   |   |   |-- ✅ py3-objarr.npy
|   |   |       |   |   |   |   |-- ✅ py3-objarr.npz
|   |   |       |   |   |   |   |-- ✅ python3.npy
|   |   |       |   |   |   |   \-- ✅ win64python2.npy
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test__datasource.py
|   |   |       |   |   |   |-- ✅ test__iotools.py
|   |   |       |   |   |   |-- ✅ test__version.py
|   |   |       |   |   |   |-- ✅ test_array_utils.py
|   |   |       |   |   |   |-- ✅ test_arraypad.py
|   |   |       |   |   |   |-- ✅ test_arraysetops.py
|   |   |       |   |   |   |-- ✅ test_arrayterator.py
|   |   |       |   |   |   |-- ✅ test_format.py
|   |   |       |   |   |   |-- ✅ test_function_base.py
|   |   |       |   |   |   |-- ✅ test_histograms.py
|   |   |       |   |   |   |-- ✅ test_index_tricks.py
|   |   |       |   |   |   |-- ✅ test_io.py
|   |   |       |   |   |   |-- ✅ test_loadtxt.py
|   |   |       |   |   |   |-- ✅ test_mixins.py
|   |   |       |   |   |   |-- ✅ test_nanfunctions.py
|   |   |       |   |   |   |-- ✅ test_packbits.py
|   |   |       |   |   |   |-- ✅ test_polynomial.py
|   |   |       |   |   |   |-- ✅ test_recfunctions.py
|   |   |       |   |   |   |-- ✅ test_regression.py
|   |   |       |   |   |   |-- ✅ test_shape_base.py
|   |   |       |   |   |   |-- ✅ test_stride_tricks.py
|   |   |       |   |   |   |-- ✅ test_twodim_base.py
|   |   |       |   |   |   |-- ✅ test_type_check.py
|   |   |       |   |   |   |-- ✅ test_ufunclike.py
|   |   |       |   |   |   \-- ✅ test_utils.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _array_utils_impl.py
|   |   |       |   |   |-- ✅ _array_utils_impl.pyi
|   |   |       |   |   |-- ✅ _arraypad_impl.py
|   |   |       |   |   |-- ✅ _arraypad_impl.pyi
|   |   |       |   |   |-- ✅ _arraysetops_impl.py
|   |   |       |   |   |-- ✅ _arraysetops_impl.pyi
|   |   |       |   |   |-- ✅ _arrayterator_impl.py
|   |   |       |   |   |-- ✅ _arrayterator_impl.pyi
|   |   |       |   |   |-- ✅ _datasource.py
|   |   |       |   |   |-- ✅ _datasource.pyi
|   |   |       |   |   |-- ✅ _format_impl.py
|   |   |       |   |   |-- ✅ _format_impl.pyi
|   |   |       |   |   |-- ✅ _function_base_impl.py
|   |   |       |   |   |-- ✅ _function_base_impl.pyi
|   |   |       |   |   |-- ✅ _histograms_impl.py
|   |   |       |   |   |-- ✅ _histograms_impl.pyi
|   |   |       |   |   |-- ✅ _index_tricks_impl.py
|   |   |       |   |   |-- ✅ _index_tricks_impl.pyi
|   |   |       |   |   |-- ✅ _iotools.py
|   |   |       |   |   |-- ✅ _iotools.pyi
|   |   |       |   |   |-- ✅ _nanfunctions_impl.py
|   |   |       |   |   |-- ✅ _nanfunctions_impl.pyi
|   |   |       |   |   |-- ✅ _npyio_impl.py
|   |   |       |   |   |-- ✅ _npyio_impl.pyi
|   |   |       |   |   |-- ✅ _polynomial_impl.py
|   |   |       |   |   |-- ✅ _polynomial_impl.pyi
|   |   |       |   |   |-- ✅ _scimath_impl.py
|   |   |       |   |   |-- ✅ _scimath_impl.pyi
|   |   |       |   |   |-- ✅ _shape_base_impl.py
|   |   |       |   |   |-- ✅ _shape_base_impl.pyi
|   |   |       |   |   |-- ✅ _stride_tricks_impl.py
|   |   |       |   |   |-- ✅ _stride_tricks_impl.pyi
|   |   |       |   |   |-- ✅ _twodim_base_impl.py
|   |   |       |   |   |-- ✅ _twodim_base_impl.pyi
|   |   |       |   |   |-- ✅ _type_check_impl.py
|   |   |       |   |   |-- ✅ _type_check_impl.pyi
|   |   |       |   |   |-- ✅ _ufunclike_impl.py
|   |   |       |   |   |-- ✅ _ufunclike_impl.pyi
|   |   |       |   |   |-- ✅ _user_array_impl.py
|   |   |       |   |   |-- ✅ _user_array_impl.pyi
|   |   |       |   |   |-- ✅ _utils_impl.py
|   |   |       |   |   |-- ✅ _utils_impl.pyi
|   |   |       |   |   |-- ✅ _version.py
|   |   |       |   |   |-- ✅ _version.pyi
|   |   |       |   |   |-- ✅ array_utils.py
|   |   |       |   |   |-- ✅ array_utils.pyi
|   |   |       |   |   |-- ✅ format.py
|   |   |       |   |   |-- ✅ format.pyi
|   |   |       |   |   |-- ✅ introspect.py
|   |   |       |   |   |-- ✅ introspect.pyi
|   |   |       |   |   |-- ✅ mixins.py
|   |   |       |   |   |-- ✅ mixins.pyi
|   |   |       |   |   |-- ✅ npyio.py
|   |   |       |   |   |-- ✅ npyio.pyi
|   |   |       |   |   |-- ✅ recfunctions.py
|   |   |       |   |   |-- ✅ recfunctions.pyi
|   |   |       |   |   |-- ✅ scimath.py
|   |   |       |   |   |-- ✅ scimath.pyi
|   |   |       |   |   |-- ✅ stride_tricks.py
|   |   |       |   |   |-- ✅ stride_tricks.pyi
|   |   |       |   |   |-- ✅ user_array.py
|   |   |       |   |   \-- ✅ user_array.pyi
|   |   |       |   |-- ✅ linalg/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_deprecations.py
|   |   |       |   |   |   |-- ✅ test_linalg.py
|   |   |       |   |   |   \-- ✅ test_regression.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _linalg.py
|   |   |       |   |   |-- ✅ _linalg.pyi
|   |   |       |   |   |-- ✅ _umath_linalg.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _umath_linalg.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _umath_linalg.pyi
|   |   |       |   |   |-- ✅ lapack_lite.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ lapack_lite.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ lapack_lite.pyi
|   |   |       |   |   |-- ✅ linalg.py
|   |   |       |   |   \-- ✅ linalg.pyi
|   |   |       |   |-- ✅ ma/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_arrayobject.py
|   |   |       |   |   |   |-- ✅ test_core.py
|   |   |       |   |   |   |-- ✅ test_deprecations.py
|   |   |       |   |   |   |-- ✅ test_extras.py
|   |   |       |   |   |   |-- ✅ test_mrecords.py
|   |   |       |   |   |   |-- ✅ test_old_ma.py
|   |   |       |   |   |   |-- ✅ test_regression.py
|   |   |       |   |   |   \-- ✅ test_subclassing.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ API_CHANGES.txt
|   |   |       |   |   |-- ✅ core.py
|   |   |       |   |   |-- ✅ core.pyi
|   |   |       |   |   |-- ✅ extras.py
|   |   |       |   |   |-- ✅ extras.pyi
|   |   |       |   |   |-- ✅ LICENSE
|   |   |       |   |   |-- ✅ mrecords.py
|   |   |       |   |   |-- ✅ mrecords.pyi
|   |   |       |   |   |-- ✅ README.rst
|   |   |       |   |   \-- ✅ testutils.py
|   |   |       |   |-- ✅ matrixlib/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_defmatrix.py
|   |   |       |   |   |   |-- ✅ test_interaction.py
|   |   |       |   |   |   |-- ✅ test_masked_matrix.py
|   |   |       |   |   |   |-- ✅ test_matrix_linalg.py
|   |   |       |   |   |   |-- ✅ test_multiarray.py
|   |   |       |   |   |   |-- ✅ test_numeric.py
|   |   |       |   |   |   \-- ✅ test_regression.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ defmatrix.py
|   |   |       |   |   \-- ✅ defmatrix.pyi
|   |   |       |   |-- ✅ polynomial/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_chebyshev.py
|   |   |       |   |   |   |-- ✅ test_classes.py
|   |   |       |   |   |   |-- ✅ test_hermite.py
|   |   |       |   |   |   |-- ✅ test_hermite_e.py
|   |   |       |   |   |   |-- ✅ test_laguerre.py
|   |   |       |   |   |   |-- ✅ test_legendre.py
|   |   |       |   |   |   |-- ✅ test_polynomial.py
|   |   |       |   |   |   |-- ✅ test_polyutils.py
|   |   |       |   |   |   |-- ✅ test_printing.py
|   |   |       |   |   |   \-- ✅ test_symbol.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _polybase.py
|   |   |       |   |   |-- ✅ _polybase.pyi
|   |   |       |   |   |-- ✅ _polytypes.pyi
|   |   |       |   |   |-- ✅ chebyshev.py
|   |   |       |   |   |-- ✅ chebyshev.pyi
|   |   |       |   |   |-- ✅ hermite.py
|   |   |       |   |   |-- ✅ hermite.pyi
|   |   |       |   |   |-- ✅ hermite_e.py
|   |   |       |   |   |-- ✅ hermite_e.pyi
|   |   |       |   |   |-- ✅ laguerre.py
|   |   |       |   |   |-- ✅ laguerre.pyi
|   |   |       |   |   |-- ✅ legendre.py
|   |   |       |   |   |-- ✅ legendre.pyi
|   |   |       |   |   |-- ✅ polynomial.py
|   |   |       |   |   |-- ✅ polynomial.pyi
|   |   |       |   |   |-- ✅ polyutils.py
|   |   |       |   |   \-- ✅ polyutils.pyi
|   |   |       |   |-- ✅ random/
|   |   |       |   |   |-- ✅ _examples/
|   |   |       |   |   |   |-- ✅ cffi/
|   |   |       |   |   |   |   |-- ✅ extending.py
|   |   |       |   |   |   |   \-- ✅ parse.py
|   |   |       |   |   |   |-- ✅ cython/
|   |   |       |   |   |   |   |-- ✅ extending.pyx
|   |   |       |   |   |   |   |-- ✅ extending_distributions.pyx
|   |   |       |   |   |   |   \-- ✅ meson.build
|   |   |       |   |   |   \-- ✅ numba/
|   |   |       |   |   |       |-- ✅ extending.py
|   |   |       |   |   |       \-- ✅ extending_distributions.py
|   |   |       |   |   |-- ✅ lib/
|   |   |       |   |   |   \-- ✅ npyrandom.lib
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ generator_pcg64_np121.pkl.gz
|   |   |       |   |   |   |   |-- ✅ generator_pcg64_np126.pkl.gz
|   |   |       |   |   |   |   |-- ✅ mt19937-testset-1.csv
|   |   |       |   |   |   |   |-- ✅ mt19937-testset-2.csv
|   |   |       |   |   |   |   |-- ✅ pcg64-testset-1.csv
|   |   |       |   |   |   |   |-- ✅ pcg64-testset-2.csv
|   |   |       |   |   |   |   |-- ✅ pcg64dxsm-testset-1.csv
|   |   |       |   |   |   |   |-- ✅ pcg64dxsm-testset-2.csv
|   |   |       |   |   |   |   |-- ✅ philox-testset-1.csv
|   |   |       |   |   |   |   |-- ✅ philox-testset-2.csv
|   |   |       |   |   |   |   |-- ✅ sfc64-testset-1.csv
|   |   |       |   |   |   |   |-- ✅ sfc64-testset-2.csv
|   |   |       |   |   |   |   \-- ✅ sfc64_np126.pkl.gz
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_direct.py
|   |   |       |   |   |   |-- ✅ test_extending.py
|   |   |       |   |   |   |-- ✅ test_generator_mt19937.py
|   |   |       |   |   |   |-- ✅ test_generator_mt19937_regressions.py
|   |   |       |   |   |   |-- ✅ test_random.py
|   |   |       |   |   |   |-- ✅ test_randomstate.py
|   |   |       |   |   |   |-- ✅ test_randomstate_regression.py
|   |   |       |   |   |   |-- ✅ test_regression.py
|   |   |       |   |   |   |-- ✅ test_seed_sequence.py
|   |   |       |   |   |   \-- ✅ test_smoke.py
|   |   |       |   |   |-- ✅ __init__.pxd
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ _bounded_integers.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _bounded_integers.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _bounded_integers.pxd
|   |   |       |   |   |-- ✅ _bounded_integers.pyi
|   |   |       |   |   |-- ✅ _common.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _common.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _common.pxd
|   |   |       |   |   |-- ✅ _common.pyi
|   |   |       |   |   |-- ✅ _generator.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _generator.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _generator.pyi
|   |   |       |   |   |-- ✅ _mt19937.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _mt19937.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _mt19937.pyi
|   |   |       |   |   |-- ✅ _pcg64.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _pcg64.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _pcg64.pyi
|   |   |       |   |   |-- ✅ _philox.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _philox.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _philox.pyi
|   |   |       |   |   |-- ✅ _pickle.py
|   |   |       |   |   |-- ✅ _pickle.pyi
|   |   |       |   |   |-- ✅ _sfc64.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _sfc64.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _sfc64.pyi
|   |   |       |   |   |-- ✅ bit_generator.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ bit_generator.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ bit_generator.pxd
|   |   |       |   |   |-- ✅ bit_generator.pyi
|   |   |       |   |   |-- ✅ c_distributions.pxd
|   |   |       |   |   |-- ✅ LICENSE.md
|   |   |       |   |   |-- ✅ mtrand.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ mtrand.cp312-win_amd64.pyd
|   |   |       |   |   \-- ✅ mtrand.pyi
|   |   |       |   |-- ✅ rec/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ __init__.pyi
|   |   |       |   |-- ✅ strings/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ __init__.pyi
|   |   |       |   |-- ✅ testing/
|   |   |       |   |   |-- ✅ _private/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |   |-- ✅ extbuild.py
|   |   |       |   |   |   |-- ✅ extbuild.pyi
|   |   |       |   |   |   |-- ✅ utils.py
|   |   |       |   |   |   \-- ✅ utils.pyi
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ test_utils.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.pyi
|   |   |       |   |   |-- ✅ overrides.py
|   |   |       |   |   |-- ✅ overrides.pyi
|   |   |       |   |   |-- ✅ print_coercion_tables.py
|   |   |       |   |   \-- ✅ print_coercion_tables.pyi
|   |   |       |   |-- ✅ tests/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ test__all__.py
|   |   |       |   |   |-- ✅ test_configtool.py
|   |   |       |   |   |-- ✅ test_ctypeslib.py
|   |   |       |   |   |-- ✅ test_lazyloading.py
|   |   |       |   |   |-- ✅ test_matlib.py
|   |   |       |   |   |-- ✅ test_numpy_config.py
|   |   |       |   |   |-- ✅ test_numpy_version.py
|   |   |       |   |   |-- ✅ test_public_api.py
|   |   |       |   |   |-- ✅ test_reloading.py
|   |   |       |   |   |-- ✅ test_scripts.py
|   |   |       |   |   \-- ✅ test_warnings.py
|   |   |       |   |-- ✅ typing/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ fail/
|   |   |       |   |   |   |   |   |-- ✅ arithmetic.pyi
|   |   |       |   |   |   |   |   |-- ✅ array_constructors.pyi
|   |   |       |   |   |   |   |   |-- ✅ array_like.pyi
|   |   |       |   |   |   |   |   |-- ✅ array_pad.pyi
|   |   |       |   |   |   |   |   |-- ✅ arrayprint.pyi
|   |   |       |   |   |   |   |   |-- ✅ arrayterator.pyi
|   |   |       |   |   |   |   |   |-- ✅ bitwise_ops.pyi
|   |   |       |   |   |   |   |   |-- ✅ char.pyi
|   |   |       |   |   |   |   |   |-- ✅ chararray.pyi
|   |   |       |   |   |   |   |   |-- ✅ comparisons.pyi
|   |   |       |   |   |   |   |   |-- ✅ constants.pyi
|   |   |       |   |   |   |   |   |-- ✅ datasource.pyi
|   |   |       |   |   |   |   |   |-- ✅ dtype.pyi
|   |   |       |   |   |   |   |   |-- ✅ einsumfunc.pyi
|   |   |       |   |   |   |   |   |-- ✅ flatiter.pyi
|   |   |       |   |   |   |   |   |-- ✅ fromnumeric.pyi
|   |   |       |   |   |   |   |   |-- ✅ histograms.pyi
|   |   |       |   |   |   |   |   |-- ✅ index_tricks.pyi
|   |   |       |   |   |   |   |   |-- ✅ lib_function_base.pyi
|   |   |       |   |   |   |   |   |-- ✅ lib_polynomial.pyi
|   |   |       |   |   |   |   |   |-- ✅ lib_utils.pyi
|   |   |       |   |   |   |   |   |-- ✅ lib_version.pyi
|   |   |       |   |   |   |   |   |-- ✅ linalg.pyi
|   |   |       |   |   |   |   |   |-- ✅ ma.pyi
|   |   |       |   |   |   |   |   |-- ✅ memmap.pyi
|   |   |       |   |   |   |   |   |-- ✅ modules.pyi
|   |   |       |   |   |   |   |   |-- ✅ multiarray.pyi
|   |   |       |   |   |   |   |   |-- ✅ ndarray.pyi
|   |   |       |   |   |   |   |   |-- ✅ ndarray_misc.pyi
|   |   |       |   |   |   |   |   |-- ✅ nditer.pyi
|   |   |       |   |   |   |   |   |-- ✅ nested_sequence.pyi
|   |   |       |   |   |   |   |   |-- ✅ npyio.pyi
|   |   |       |   |   |   |   |   |-- ✅ numerictypes.pyi
|   |   |       |   |   |   |   |   |-- ✅ random.pyi
|   |   |       |   |   |   |   |   |-- ✅ rec.pyi
|   |   |       |   |   |   |   |   |-- ✅ scalars.pyi
|   |   |       |   |   |   |   |   |-- ✅ shape.pyi
|   |   |       |   |   |   |   |   |-- ✅ shape_base.pyi
|   |   |       |   |   |   |   |   |-- ✅ stride_tricks.pyi
|   |   |       |   |   |   |   |   |-- ✅ strings.pyi
|   |   |       |   |   |   |   |   |-- ✅ testing.pyi
|   |   |       |   |   |   |   |   |-- ✅ twodim_base.pyi
|   |   |       |   |   |   |   |   |-- ✅ type_check.pyi
|   |   |       |   |   |   |   |   |-- ✅ ufunc_config.pyi
|   |   |       |   |   |   |   |   |-- ✅ ufunclike.pyi
|   |   |       |   |   |   |   |   |-- ✅ ufuncs.pyi
|   |   |       |   |   |   |   |   \-- ✅ warnings_and_errors.pyi
|   |   |       |   |   |   |   |-- ✅ misc/
|   |   |       |   |   |   |   |   \-- ✅ extended_precision.pyi
|   |   |       |   |   |   |   |-- ✅ pass/
|   |   |       |   |   |   |   |   |-- ✅ arithmetic.py
|   |   |       |   |   |   |   |   |-- ✅ array_constructors.py
|   |   |       |   |   |   |   |   |-- ✅ array_like.py
|   |   |       |   |   |   |   |   |-- ✅ arrayprint.py
|   |   |       |   |   |   |   |   |-- ✅ arrayterator.py
|   |   |       |   |   |   |   |   |-- ✅ bitwise_ops.py
|   |   |       |   |   |   |   |   |-- ✅ comparisons.py
|   |   |       |   |   |   |   |   |-- ✅ dtype.py
|   |   |       |   |   |   |   |   |-- ✅ einsumfunc.py
|   |   |       |   |   |   |   |   |-- ✅ flatiter.py
|   |   |       |   |   |   |   |   |-- ✅ fromnumeric.py
|   |   |       |   |   |   |   |   |-- ✅ index_tricks.py
|   |   |       |   |   |   |   |   |-- ✅ lib_user_array.py
|   |   |       |   |   |   |   |   |-- ✅ lib_utils.py
|   |   |       |   |   |   |   |   |-- ✅ lib_version.py
|   |   |       |   |   |   |   |   |-- ✅ literal.py
|   |   |       |   |   |   |   |   |-- ✅ ma.py
|   |   |       |   |   |   |   |   |-- ✅ mod.py
|   |   |       |   |   |   |   |   |-- ✅ modules.py
|   |   |       |   |   |   |   |   |-- ✅ multiarray.py
|   |   |       |   |   |   |   |   |-- ✅ ndarray_conversion.py
|   |   |       |   |   |   |   |   |-- ✅ ndarray_misc.py
|   |   |       |   |   |   |   |   |-- ✅ ndarray_shape_manipulation.py
|   |   |       |   |   |   |   |   |-- ✅ nditer.py
|   |   |       |   |   |   |   |   |-- ✅ numeric.py
|   |   |       |   |   |   |   |   |-- ✅ numerictypes.py
|   |   |       |   |   |   |   |   |-- ✅ random.py
|   |   |       |   |   |   |   |   |-- ✅ recfunctions.py
|   |   |       |   |   |   |   |   |-- ✅ scalars.py
|   |   |       |   |   |   |   |   |-- ✅ shape.py
|   |   |       |   |   |   |   |   |-- ✅ simple.py
|   |   |       |   |   |   |   |   |-- ✅ simple_py3.py
|   |   |       |   |   |   |   |   |-- ✅ ufunc_config.py
|   |   |       |   |   |   |   |   |-- ✅ ufunclike.py
|   |   |       |   |   |   |   |   |-- ✅ ufuncs.py
|   |   |       |   |   |   |   |   \-- ✅ warnings_and_errors.py
|   |   |       |   |   |   |   |-- ✅ reveal/
|   |   |       |   |   |   |   |   |-- ✅ arithmetic.pyi
|   |   |       |   |   |   |   |   |-- ✅ array_api_info.pyi
|   |   |       |   |   |   |   |   |-- ✅ array_constructors.pyi
|   |   |       |   |   |   |   |   |-- ✅ arraypad.pyi
|   |   |       |   |   |   |   |   |-- ✅ arrayprint.pyi
|   |   |       |   |   |   |   |   |-- ✅ arraysetops.pyi
|   |   |       |   |   |   |   |   |-- ✅ arrayterator.pyi
|   |   |       |   |   |   |   |   |-- ✅ bitwise_ops.pyi
|   |   |       |   |   |   |   |   |-- ✅ char.pyi
|   |   |       |   |   |   |   |   |-- ✅ chararray.pyi
|   |   |       |   |   |   |   |   |-- ✅ comparisons.pyi
|   |   |       |   |   |   |   |   |-- ✅ constants.pyi
|   |   |       |   |   |   |   |   |-- ✅ ctypeslib.pyi
|   |   |       |   |   |   |   |   |-- ✅ datasource.pyi
|   |   |       |   |   |   |   |   |-- ✅ dtype.pyi
|   |   |       |   |   |   |   |   |-- ✅ einsumfunc.pyi
|   |   |       |   |   |   |   |   |-- ✅ emath.pyi
|   |   |       |   |   |   |   |   |-- ✅ fft.pyi
|   |   |       |   |   |   |   |   |-- ✅ flatiter.pyi
|   |   |       |   |   |   |   |   |-- ✅ fromnumeric.pyi
|   |   |       |   |   |   |   |   |-- ✅ getlimits.pyi
|   |   |       |   |   |   |   |   |-- ✅ histograms.pyi
|   |   |       |   |   |   |   |   |-- ✅ index_tricks.pyi
|   |   |       |   |   |   |   |   |-- ✅ lib_function_base.pyi
|   |   |       |   |   |   |   |   |-- ✅ lib_polynomial.pyi
|   |   |       |   |   |   |   |   |-- ✅ lib_utils.pyi
|   |   |       |   |   |   |   |   |-- ✅ lib_version.pyi
|   |   |       |   |   |   |   |   |-- ✅ linalg.pyi
|   |   |       |   |   |   |   |   |-- ✅ ma.pyi
|   |   |       |   |   |   |   |   |-- ✅ matrix.pyi
|   |   |       |   |   |   |   |   |-- ✅ memmap.pyi
|   |   |       |   |   |   |   |   |-- ✅ mod.pyi
|   |   |       |   |   |   |   |   |-- ✅ modules.pyi
|   |   |       |   |   |   |   |   |-- ✅ multiarray.pyi
|   |   |       |   |   |   |   |   |-- ✅ nbit_base_example.pyi
|   |   |       |   |   |   |   |   |-- ✅ ndarray_assignability.pyi
|   |   |       |   |   |   |   |   |-- ✅ ndarray_conversion.pyi
|   |   |       |   |   |   |   |   |-- ✅ ndarray_misc.pyi
|   |   |       |   |   |   |   |   |-- ✅ ndarray_shape_manipulation.pyi
|   |   |       |   |   |   |   |   |-- ✅ nditer.pyi
|   |   |       |   |   |   |   |   |-- ✅ nested_sequence.pyi
|   |   |       |   |   |   |   |   |-- ✅ npyio.pyi
|   |   |       |   |   |   |   |   |-- ✅ numeric.pyi
|   |   |       |   |   |   |   |   |-- ✅ numerictypes.pyi
|   |   |       |   |   |   |   |   |-- ✅ polynomial_polybase.pyi
|   |   |       |   |   |   |   |   |-- ✅ polynomial_polyutils.pyi
|   |   |       |   |   |   |   |   |-- ✅ polynomial_series.pyi
|   |   |       |   |   |   |   |   |-- ✅ random.pyi
|   |   |       |   |   |   |   |   |-- ✅ rec.pyi
|   |   |       |   |   |   |   |   |-- ✅ scalars.pyi
|   |   |       |   |   |   |   |   |-- ✅ shape.pyi
|   |   |       |   |   |   |   |   |-- ✅ shape_base.pyi
|   |   |       |   |   |   |   |   |-- ✅ stride_tricks.pyi
|   |   |       |   |   |   |   |   |-- ✅ strings.pyi
|   |   |       |   |   |   |   |   |-- ✅ testing.pyi
|   |   |       |   |   |   |   |   |-- ✅ twodim_base.pyi
|   |   |       |   |   |   |   |   |-- ✅ type_check.pyi
|   |   |       |   |   |   |   |   |-- ✅ ufunc_config.pyi
|   |   |       |   |   |   |   |   |-- ✅ ufunclike.pyi
|   |   |       |   |   |   |   |   |-- ✅ ufuncs.pyi
|   |   |       |   |   |   |   |   \-- ✅ warnings_and_errors.pyi
|   |   |       |   |   |   |   \-- ✅ mypy.ini
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_isfile.py
|   |   |       |   |   |   |-- ✅ test_runtime.py
|   |   |       |   |   |   \-- ✅ test_typing.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ mypy_plugin.py
|   |   |       |   |-- ✅ __config__.py
|   |   |       |   |-- ✅ __config__.pyi
|   |   |       |   |-- ✅ __init__.cython-30.pxd
|   |   |       |   |-- ✅ __init__.pxd
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __init__.pyi
|   |   |       |   |-- ✅ _array_api_info.py
|   |   |       |   |-- ✅ _array_api_info.pyi
|   |   |       |   |-- ✅ _configtool.py
|   |   |       |   |-- ✅ _configtool.pyi
|   |   |       |   |-- ✅ _distributor_init.py
|   |   |       |   |-- ✅ _distributor_init.pyi
|   |   |       |   |-- ✅ _expired_attrs_2_0.py
|   |   |       |   |-- ✅ _expired_attrs_2_0.pyi
|   |   |       |   |-- ✅ _globals.py
|   |   |       |   |-- ✅ _globals.pyi
|   |   |       |   |-- ✅ _pytesttester.py
|   |   |       |   |-- ✅ _pytesttester.pyi
|   |   |       |   |-- ✅ conftest.py
|   |   |       |   |-- ✅ dtypes.py
|   |   |       |   |-- ✅ dtypes.pyi
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ exceptions.pyi
|   |   |       |   |-- ✅ matlib.py
|   |   |       |   |-- ✅ matlib.pyi
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ version.py
|   |   |       |   \-- ✅ version.pyi
|   |   |       |-- ✅ numpy-2.3.3.dist-info/
|   |   |       |   |-- ✅ DELVEWHEEL
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ numpy.libs/
|   |   |       |   |-- ✅ libscipy_openblas64_-860d95b1c38e637ce4509f5fa24fbf2a.dll
|   |   |       |   \-- ✅ msvcp140-a4c2229bdc2a2a630acdc095b4d86008.dll
|   |   |       |-- ✅ packaging/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ _spdx.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _elffile.py
|   |   |       |   |-- ✅ _manylinux.py
|   |   |       |   |-- ✅ _musllinux.py
|   |   |       |   |-- ✅ _parser.py
|   |   |       |   |-- ✅ _structures.py
|   |   |       |   |-- ✅ _tokenizer.py
|   |   |       |   |-- ✅ markers.py
|   |   |       |   |-- ✅ metadata.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ requirements.py
|   |   |       |   |-- ✅ specifiers.py
|   |   |       |   |-- ✅ tags.py
|   |   |       |   |-- ✅ utils.py
|   |   |       |   \-- ✅ version.py
|   |   |       |-- ✅ packaging-25.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ LICENSE
|   |   |       |   |   |-- ✅ LICENSE.APACHE
|   |   |       |   |   \-- ✅ LICENSE.BSD
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ pandas/
|   |   |       |   |-- ✅ _config/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ config.py
|   |   |       |   |   |-- ✅ dates.py
|   |   |       |   |   |-- ✅ display.py
|   |   |       |   |   \-- ✅ localization.py
|   |   |       |   |-- ✅ _libs/
|   |   |       |   |   |-- ✅ tslibs/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ base.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ base.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ ccalendar.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ ccalendar.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ ccalendar.pyi
|   |   |       |   |   |   |-- ✅ conversion.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ conversion.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ conversion.pyi
|   |   |       |   |   |   |-- ✅ dtypes.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ dtypes.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ dtypes.pyi
|   |   |       |   |   |   |-- ✅ fields.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ fields.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ fields.pyi
|   |   |       |   |   |   |-- ✅ nattype.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ nattype.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ nattype.pyi
|   |   |       |   |   |   |-- ✅ np_datetime.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ np_datetime.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ np_datetime.pyi
|   |   |       |   |   |   |-- ✅ offsets.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ offsets.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ offsets.pyi
|   |   |       |   |   |   |-- ✅ parsing.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ parsing.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ parsing.pyi
|   |   |       |   |   |   |-- ✅ period.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ period.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ period.pyi
|   |   |       |   |   |   |-- ✅ strptime.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ strptime.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ strptime.pyi
|   |   |       |   |   |   |-- ✅ timedeltas.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ timedeltas.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ timedeltas.pyi
|   |   |       |   |   |   |-- ✅ timestamps.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ timestamps.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ timestamps.pyi
|   |   |       |   |   |   |-- ✅ timezones.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ timezones.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ timezones.pyi
|   |   |       |   |   |   |-- ✅ tzconversion.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ tzconversion.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ tzconversion.pyi
|   |   |       |   |   |   |-- ✅ vectorized.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ vectorized.cp312-win_amd64.pyd
|   |   |       |   |   |   \-- ✅ vectorized.pyi
|   |   |       |   |   |-- ✅ window/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ aggregations.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ aggregations.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ aggregations.pyi
|   |   |       |   |   |   |-- ✅ indexers.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ indexers.cp312-win_amd64.pyd
|   |   |       |   |   |   \-- ✅ indexers.pyi
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ algos.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ algos.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ algos.pyi
|   |   |       |   |   |-- ✅ arrays.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ arrays.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ arrays.pyi
|   |   |       |   |   |-- ✅ byteswap.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ byteswap.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ byteswap.pyi
|   |   |       |   |   |-- ✅ groupby.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ groupby.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ groupby.pyi
|   |   |       |   |   |-- ✅ hashing.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ hashing.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ hashing.pyi
|   |   |       |   |   |-- ✅ hashtable.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ hashtable.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ hashtable.pyi
|   |   |       |   |   |-- ✅ index.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ index.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ index.pyi
|   |   |       |   |   |-- ✅ indexing.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ indexing.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ indexing.pyi
|   |   |       |   |   |-- ✅ internals.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ internals.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ internals.pyi
|   |   |       |   |   |-- ✅ interval.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ interval.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ interval.pyi
|   |   |       |   |   |-- ✅ join.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ join.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ join.pyi
|   |   |       |   |   |-- ✅ json.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ json.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ json.pyi
|   |   |       |   |   |-- ✅ lib.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ lib.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ lib.pyi
|   |   |       |   |   |-- ✅ missing.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ missing.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ missing.pyi
|   |   |       |   |   |-- ✅ ops.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ ops.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ ops.pyi
|   |   |       |   |   |-- ✅ ops_dispatch.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ ops_dispatch.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ ops_dispatch.pyi
|   |   |       |   |   |-- ✅ pandas_datetime.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ pandas_datetime.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ pandas_parser.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ pandas_parser.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ parsers.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ parsers.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ parsers.pyi
|   |   |       |   |   |-- ✅ properties.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ properties.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ properties.pyi
|   |   |       |   |   |-- ✅ reshape.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ reshape.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ reshape.pyi
|   |   |       |   |   |-- ✅ sas.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ sas.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ sas.pyi
|   |   |       |   |   |-- ✅ sparse.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ sparse.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ sparse.pyi
|   |   |       |   |   |-- ✅ testing.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ testing.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ testing.pyi
|   |   |       |   |   |-- ✅ tslib.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ tslib.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ tslib.pyi
|   |   |       |   |   |-- ✅ writers.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ writers.cp312-win_amd64.pyd
|   |   |       |   |   \-- ✅ writers.pyi
|   |   |       |   |-- ✅ _testing/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _hypothesis.py
|   |   |       |   |   |-- ✅ _io.py
|   |   |       |   |   |-- ✅ _warnings.py
|   |   |       |   |   |-- ✅ asserters.py
|   |   |       |   |   |-- ✅ compat.py
|   |   |       |   |   \-- ✅ contexts.py
|   |   |       |   |-- ✅ api/
|   |   |       |   |   |-- ✅ extensions/
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ indexers/
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ interchange/
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ types/
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ typing/
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   \-- ✅ __init__.py
|   |   |       |   |-- ✅ arrays/
|   |   |       |   |   \-- ✅ __init__.py
|   |   |       |   |-- ✅ compat/
|   |   |       |   |   |-- ✅ numpy/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ function.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _constants.py
|   |   |       |   |   |-- ✅ _optional.py
|   |   |       |   |   |-- ✅ compressors.py
|   |   |       |   |   |-- ✅ pickle_compat.py
|   |   |       |   |   \-- ✅ pyarrow.py
|   |   |       |   |-- ✅ core/
|   |   |       |   |   |-- ✅ _numba/
|   |   |       |   |   |   |-- ✅ kernels/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ mean_.py
|   |   |       |   |   |   |   |-- ✅ min_max_.py
|   |   |       |   |   |   |   |-- ✅ shared.py
|   |   |       |   |   |   |   |-- ✅ sum_.py
|   |   |       |   |   |   |   \-- ✅ var_.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ executor.py
|   |   |       |   |   |   \-- ✅ extensions.py
|   |   |       |   |   |-- ✅ array_algos/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ datetimelike_accumulations.py
|   |   |       |   |   |   |-- ✅ masked_accumulations.py
|   |   |       |   |   |   |-- ✅ masked_reductions.py
|   |   |       |   |   |   |-- ✅ putmask.py
|   |   |       |   |   |   |-- ✅ quantile.py
|   |   |       |   |   |   |-- ✅ replace.py
|   |   |       |   |   |   |-- ✅ take.py
|   |   |       |   |   |   \-- ✅ transforms.py
|   |   |       |   |   |-- ✅ arrays/
|   |   |       |   |   |   |-- ✅ arrow/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _arrow_utils.py
|   |   |       |   |   |   |   |-- ✅ accessors.py
|   |   |       |   |   |   |   |-- ✅ array.py
|   |   |       |   |   |   |   \-- ✅ extension_types.py
|   |   |       |   |   |   |-- ✅ sparse/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ accessor.py
|   |   |       |   |   |   |   |-- ✅ array.py
|   |   |       |   |   |   |   \-- ✅ scipy_sparse.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _arrow_string_mixins.py
|   |   |       |   |   |   |-- ✅ _mixins.py
|   |   |       |   |   |   |-- ✅ _ranges.py
|   |   |       |   |   |   |-- ✅ _utils.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ boolean.py
|   |   |       |   |   |   |-- ✅ categorical.py
|   |   |       |   |   |   |-- ✅ datetimelike.py
|   |   |       |   |   |   |-- ✅ datetimes.py
|   |   |       |   |   |   |-- ✅ floating.py
|   |   |       |   |   |   |-- ✅ integer.py
|   |   |       |   |   |   |-- ✅ interval.py
|   |   |       |   |   |   |-- ✅ masked.py
|   |   |       |   |   |   |-- ✅ numeric.py
|   |   |       |   |   |   |-- ✅ numpy_.py
|   |   |       |   |   |   |-- ✅ period.py
|   |   |       |   |   |   |-- ✅ string_.py
|   |   |       |   |   |   |-- ✅ string_arrow.py
|   |   |       |   |   |   \-- ✅ timedeltas.py
|   |   |       |   |   |-- ✅ computation/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ align.py
|   |   |       |   |   |   |-- ✅ api.py
|   |   |       |   |   |   |-- ✅ check.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ engines.py
|   |   |       |   |   |   |-- ✅ eval.py
|   |   |       |   |   |   |-- ✅ expr.py
|   |   |       |   |   |   |-- ✅ expressions.py
|   |   |       |   |   |   |-- ✅ ops.py
|   |   |       |   |   |   |-- ✅ parsing.py
|   |   |       |   |   |   |-- ✅ pytables.py
|   |   |       |   |   |   \-- ✅ scope.py
|   |   |       |   |   |-- ✅ dtypes/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ api.py
|   |   |       |   |   |   |-- ✅ astype.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ cast.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ concat.py
|   |   |       |   |   |   |-- ✅ dtypes.py
|   |   |       |   |   |   |-- ✅ generic.py
|   |   |       |   |   |   |-- ✅ inference.py
|   |   |       |   |   |   \-- ✅ missing.py
|   |   |       |   |   |-- ✅ groupby/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ categorical.py
|   |   |       |   |   |   |-- ✅ generic.py
|   |   |       |   |   |   |-- ✅ groupby.py
|   |   |       |   |   |   |-- ✅ grouper.py
|   |   |       |   |   |   |-- ✅ indexing.py
|   |   |       |   |   |   |-- ✅ numba_.py
|   |   |       |   |   |   \-- ✅ ops.py
|   |   |       |   |   |-- ✅ indexers/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ objects.py
|   |   |       |   |   |   \-- ✅ utils.py
|   |   |       |   |   |-- ✅ indexes/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ accessors.py
|   |   |       |   |   |   |-- ✅ api.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ category.py
|   |   |       |   |   |   |-- ✅ datetimelike.py
|   |   |       |   |   |   |-- ✅ datetimes.py
|   |   |       |   |   |   |-- ✅ extension.py
|   |   |       |   |   |   |-- ✅ frozen.py
|   |   |       |   |   |   |-- ✅ interval.py
|   |   |       |   |   |   |-- ✅ multi.py
|   |   |       |   |   |   |-- ✅ period.py
|   |   |       |   |   |   |-- ✅ range.py
|   |   |       |   |   |   \-- ✅ timedeltas.py
|   |   |       |   |   |-- ✅ interchange/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ buffer.py
|   |   |       |   |   |   |-- ✅ column.py
|   |   |       |   |   |   |-- ✅ dataframe.py
|   |   |       |   |   |   |-- ✅ dataframe_protocol.py
|   |   |       |   |   |   |-- ✅ from_dataframe.py
|   |   |       |   |   |   \-- ✅ utils.py
|   |   |       |   |   |-- ✅ internals/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ api.py
|   |   |       |   |   |   |-- ✅ array_manager.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ blocks.py
|   |   |       |   |   |   |-- ✅ concat.py
|   |   |       |   |   |   |-- ✅ construction.py
|   |   |       |   |   |   |-- ✅ managers.py
|   |   |       |   |   |   \-- ✅ ops.py
|   |   |       |   |   |-- ✅ methods/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ describe.py
|   |   |       |   |   |   |-- ✅ selectn.py
|   |   |       |   |   |   \-- ✅ to_dict.py
|   |   |       |   |   |-- ✅ ops/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ array_ops.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ dispatch.py
|   |   |       |   |   |   |-- ✅ docstrings.py
|   |   |       |   |   |   |-- ✅ invalid.py
|   |   |       |   |   |   |-- ✅ mask_ops.py
|   |   |       |   |   |   \-- ✅ missing.py
|   |   |       |   |   |-- ✅ reshape/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ api.py
|   |   |       |   |   |   |-- ✅ concat.py
|   |   |       |   |   |   |-- ✅ encoding.py
|   |   |       |   |   |   |-- ✅ melt.py
|   |   |       |   |   |   |-- ✅ merge.py
|   |   |       |   |   |   |-- ✅ pivot.py
|   |   |       |   |   |   |-- ✅ reshape.py
|   |   |       |   |   |   |-- ✅ tile.py
|   |   |       |   |   |   \-- ✅ util.py
|   |   |       |   |   |-- ✅ sparse/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ api.py
|   |   |       |   |   |-- ✅ strings/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ accessor.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   \-- ✅ object_array.py
|   |   |       |   |   |-- ✅ tools/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ datetimes.py
|   |   |       |   |   |   |-- ✅ numeric.py
|   |   |       |   |   |   |-- ✅ timedeltas.py
|   |   |       |   |   |   \-- ✅ times.py
|   |   |       |   |   |-- ✅ util/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ hashing.py
|   |   |       |   |   |   \-- ✅ numba_.py
|   |   |       |   |   |-- ✅ window/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ doc.py
|   |   |       |   |   |   |-- ✅ ewm.py
|   |   |       |   |   |   |-- ✅ expanding.py
|   |   |       |   |   |   |-- ✅ numba_.py
|   |   |       |   |   |   |-- ✅ online.py
|   |   |       |   |   |   \-- ✅ rolling.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ accessor.py
|   |   |       |   |   |-- ✅ algorithms.py
|   |   |       |   |   |-- ✅ api.py
|   |   |       |   |   |-- ✅ apply.py
|   |   |       |   |   |-- ✅ arraylike.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ common.py
|   |   |       |   |   |-- ✅ config_init.py
|   |   |       |   |   |-- ✅ construction.py
|   |   |       |   |   |-- ✅ flags.py
|   |   |       |   |   |-- ✅ frame.py
|   |   |       |   |   |-- ✅ generic.py
|   |   |       |   |   |-- ✅ indexing.py
|   |   |       |   |   |-- ✅ missing.py
|   |   |       |   |   |-- ✅ nanops.py
|   |   |       |   |   |-- ✅ resample.py
|   |   |       |   |   |-- ✅ roperator.py
|   |   |       |   |   |-- ✅ sample.py
|   |   |       |   |   |-- ✅ series.py
|   |   |       |   |   |-- ✅ shared_docs.py
|   |   |       |   |   \-- ✅ sorting.py
|   |   |       |   |-- ✅ errors/
|   |   |       |   |   \-- ✅ __init__.py
|   |   |       |   |-- ✅ io/
|   |   |       |   |   |-- ✅ clipboard/
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ excel/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _base.py
|   |   |       |   |   |   |-- ✅ _calamine.py
|   |   |       |   |   |   |-- ✅ _odfreader.py
|   |   |       |   |   |   |-- ✅ _odswriter.py
|   |   |       |   |   |   |-- ✅ _openpyxl.py
|   |   |       |   |   |   |-- ✅ _pyxlsb.py
|   |   |       |   |   |   |-- ✅ _util.py
|   |   |       |   |   |   |-- ✅ _xlrd.py
|   |   |       |   |   |   \-- ✅ _xlsxwriter.py
|   |   |       |   |   |-- ✅ formats/
|   |   |       |   |   |   |-- ✅ templates/
|   |   |       |   |   |   |   |-- ✅ html.tpl
|   |   |       |   |   |   |   |-- ✅ html_style.tpl
|   |   |       |   |   |   |   |-- ✅ html_table.tpl
|   |   |       |   |   |   |   |-- ✅ latex.tpl
|   |   |       |   |   |   |   |-- ✅ latex_longtable.tpl
|   |   |       |   |   |   |   |-- ✅ latex_table.tpl
|   |   |       |   |   |   |   \-- ✅ string.tpl
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _color_data.py
|   |   |       |   |   |   |-- ✅ console.py
|   |   |       |   |   |   |-- ✅ css.py
|   |   |       |   |   |   |-- ✅ csvs.py
|   |   |       |   |   |   |-- ✅ excel.py
|   |   |       |   |   |   |-- ✅ format.py
|   |   |       |   |   |   |-- ✅ html.py
|   |   |       |   |   |   |-- ✅ info.py
|   |   |       |   |   |   |-- ✅ printing.py
|   |   |       |   |   |   |-- ✅ string.py
|   |   |       |   |   |   |-- ✅ style.py
|   |   |       |   |   |   |-- ✅ style_render.py
|   |   |       |   |   |   \-- ✅ xml.py
|   |   |       |   |   |-- ✅ json/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _json.py
|   |   |       |   |   |   |-- ✅ _normalize.py
|   |   |       |   |   |   \-- ✅ _table_schema.py
|   |   |       |   |   |-- ✅ parsers/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ arrow_parser_wrapper.py
|   |   |       |   |   |   |-- ✅ base_parser.py
|   |   |       |   |   |   |-- ✅ c_parser_wrapper.py
|   |   |       |   |   |   |-- ✅ python_parser.py
|   |   |       |   |   |   \-- ✅ readers.py
|   |   |       |   |   |-- ✅ sas/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ sas7bdat.py
|   |   |       |   |   |   |-- ✅ sas_constants.py
|   |   |       |   |   |   |-- ✅ sas_xport.py
|   |   |       |   |   |   \-- ✅ sasreader.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _util.py
|   |   |       |   |   |-- ✅ api.py
|   |   |       |   |   |-- ✅ clipboards.py
|   |   |       |   |   |-- ✅ common.py
|   |   |       |   |   |-- ✅ feather_format.py
|   |   |       |   |   |-- ✅ gbq.py
|   |   |       |   |   |-- ✅ html.py
|   |   |       |   |   |-- ✅ orc.py
|   |   |       |   |   |-- ✅ parquet.py
|   |   |       |   |   |-- ✅ pickle.py
|   |   |       |   |   |-- ✅ pytables.py
|   |   |       |   |   |-- ✅ spss.py
|   |   |       |   |   |-- ✅ sql.py
|   |   |       |   |   |-- ✅ stata.py
|   |   |       |   |   \-- ✅ xml.py
|   |   |       |   |-- ✅ plotting/
|   |   |       |   |   |-- ✅ _matplotlib/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ boxplot.py
|   |   |       |   |   |   |-- ✅ converter.py
|   |   |       |   |   |   |-- ✅ core.py
|   |   |       |   |   |   |-- ✅ groupby.py
|   |   |       |   |   |   |-- ✅ hist.py
|   |   |       |   |   |   |-- ✅ misc.py
|   |   |       |   |   |   |-- ✅ style.py
|   |   |       |   |   |   |-- ✅ timeseries.py
|   |   |       |   |   |   \-- ✅ tools.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _core.py
|   |   |       |   |   \-- ✅ _misc.py
|   |   |       |   |-- ✅ tests/
|   |   |       |   |   |-- ✅ api/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_api.py
|   |   |       |   |   |   \-- ✅ test_types.py
|   |   |       |   |   |-- ✅ apply/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ test_frame_apply.py
|   |   |       |   |   |   |-- ✅ test_frame_apply_relabeling.py
|   |   |       |   |   |   |-- ✅ test_frame_transform.py
|   |   |       |   |   |   |-- ✅ test_invalid_arg.py
|   |   |       |   |   |   |-- ✅ test_numba.py
|   |   |       |   |   |   |-- ✅ test_series_apply.py
|   |   |       |   |   |   |-- ✅ test_series_apply_relabeling.py
|   |   |       |   |   |   |-- ✅ test_series_transform.py
|   |   |       |   |   |   \-- ✅ test_str.py
|   |   |       |   |   |-- ✅ arithmetic/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_array_ops.py
|   |   |       |   |   |   |-- ✅ test_categorical.py
|   |   |       |   |   |   |-- ✅ test_datetime64.py
|   |   |       |   |   |   |-- ✅ test_interval.py
|   |   |       |   |   |   |-- ✅ test_numeric.py
|   |   |       |   |   |   |-- ✅ test_object.py
|   |   |       |   |   |   |-- ✅ test_period.py
|   |   |       |   |   |   \-- ✅ test_timedelta64.py
|   |   |       |   |   |-- ✅ arrays/
|   |   |       |   |   |   |-- ✅ boolean/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_comparison.py
|   |   |       |   |   |   |   |-- ✅ test_construction.py
|   |   |       |   |   |   |   |-- ✅ test_function.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_logical.py
|   |   |       |   |   |   |   |-- ✅ test_ops.py
|   |   |       |   |   |   |   |-- ✅ test_reduction.py
|   |   |       |   |   |   |   \-- ✅ test_repr.py
|   |   |       |   |   |   |-- ✅ categorical/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_algos.py
|   |   |       |   |   |   |   |-- ✅ test_analytics.py
|   |   |       |   |   |   |   |-- ✅ test_api.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_dtypes.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_map.py
|   |   |       |   |   |   |   |-- ✅ test_missing.py
|   |   |       |   |   |   |   |-- ✅ test_operators.py
|   |   |       |   |   |   |   |-- ✅ test_replace.py
|   |   |       |   |   |   |   |-- ✅ test_repr.py
|   |   |       |   |   |   |   |-- ✅ test_sorting.py
|   |   |       |   |   |   |   |-- ✅ test_subclass.py
|   |   |       |   |   |   |   |-- ✅ test_take.py
|   |   |       |   |   |   |   \-- ✅ test_warnings.py
|   |   |       |   |   |   |-- ✅ datetimes/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_cumulative.py
|   |   |       |   |   |   |   \-- ✅ test_reductions.py
|   |   |       |   |   |   |-- ✅ floating/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_comparison.py
|   |   |       |   |   |   |   |-- ✅ test_concat.py
|   |   |       |   |   |   |   |-- ✅ test_construction.py
|   |   |       |   |   |   |   |-- ✅ test_contains.py
|   |   |       |   |   |   |   |-- ✅ test_function.py
|   |   |       |   |   |   |   |-- ✅ test_repr.py
|   |   |       |   |   |   |   \-- ✅ test_to_numpy.py
|   |   |       |   |   |   |-- ✅ integer/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |   |-- ✅ test_comparison.py
|   |   |       |   |   |   |   |-- ✅ test_concat.py
|   |   |       |   |   |   |   |-- ✅ test_construction.py
|   |   |       |   |   |   |   |-- ✅ test_dtypes.py
|   |   |       |   |   |   |   |-- ✅ test_function.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_reduction.py
|   |   |       |   |   |   |   \-- ✅ test_repr.py
|   |   |       |   |   |   |-- ✅ interval/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |   |-- ✅ test_interval.py
|   |   |       |   |   |   |   |-- ✅ test_interval_pyarrow.py
|   |   |       |   |   |   |   \-- ✅ test_overlaps.py
|   |   |       |   |   |   |-- ✅ masked/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |   |-- ✅ test_arrow_compat.py
|   |   |       |   |   |   |   |-- ✅ test_function.py
|   |   |       |   |   |   |   \-- ✅ test_indexing.py
|   |   |       |   |   |   |-- ✅ numpy_/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   \-- ✅ test_numpy.py
|   |   |       |   |   |   |-- ✅ period/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_arrow_compat.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   \-- ✅ test_reductions.py
|   |   |       |   |   |   |-- ✅ sparse/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_accessor.py
|   |   |       |   |   |   |   |-- ✅ test_arithmetics.py
|   |   |       |   |   |   |   |-- ✅ test_array.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_combine_concat.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_dtype.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_libsparse.py
|   |   |       |   |   |   |   |-- ✅ test_reductions.py
|   |   |       |   |   |   |   \-- ✅ test_unary.py
|   |   |       |   |   |   |-- ✅ string_/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_concat.py
|   |   |       |   |   |   |   |-- ✅ test_string.py
|   |   |       |   |   |   |   \-- ✅ test_string_arrow.py
|   |   |       |   |   |   |-- ✅ timedeltas/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_cumulative.py
|   |   |       |   |   |   |   \-- ✅ test_reductions.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ masked_shared.py
|   |   |       |   |   |   |-- ✅ test_array.py
|   |   |       |   |   |   |-- ✅ test_datetimelike.py
|   |   |       |   |   |   |-- ✅ test_datetimes.py
|   |   |       |   |   |   |-- ✅ test_ndarray_backed.py
|   |   |       |   |   |   |-- ✅ test_period.py
|   |   |       |   |   |   \-- ✅ test_timedeltas.py
|   |   |       |   |   |-- ✅ base/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |-- ✅ test_conversion.py
|   |   |       |   |   |   |-- ✅ test_fillna.py
|   |   |       |   |   |   |-- ✅ test_misc.py
|   |   |       |   |   |   |-- ✅ test_transpose.py
|   |   |       |   |   |   |-- ✅ test_unique.py
|   |   |       |   |   |   \-- ✅ test_value_counts.py
|   |   |       |   |   |-- ✅ computation/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_compat.py
|   |   |       |   |   |   \-- ✅ test_eval.py
|   |   |       |   |   |-- ✅ config/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_config.py
|   |   |       |   |   |   \-- ✅ test_localization.py
|   |   |       |   |   |-- ✅ construction/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ test_extract_array.py
|   |   |       |   |   |-- ✅ copy_view/
|   |   |       |   |   |   |-- ✅ index/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_datetimeindex.py
|   |   |       |   |   |   |   |-- ✅ test_index.py
|   |   |       |   |   |   |   |-- ✅ test_periodindex.py
|   |   |       |   |   |   |   \-- ✅ test_timedeltaindex.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_array.py
|   |   |       |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |-- ✅ test_chained_assignment_deprecation.py
|   |   |       |   |   |   |-- ✅ test_clip.py
|   |   |       |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |-- ✅ test_core_functionalities.py
|   |   |       |   |   |   |-- ✅ test_functions.py
|   |   |       |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |-- ✅ test_internals.py
|   |   |       |   |   |   |-- ✅ test_interp_fillna.py
|   |   |       |   |   |   |-- ✅ test_methods.py
|   |   |       |   |   |   |-- ✅ test_replace.py
|   |   |       |   |   |   |-- ✅ test_setitem.py
|   |   |       |   |   |   |-- ✅ test_util.py
|   |   |       |   |   |   \-- ✅ util.py
|   |   |       |   |   |-- ✅ dtypes/
|   |   |       |   |   |   |-- ✅ cast/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_can_hold_element.py
|   |   |       |   |   |   |   |-- ✅ test_construct_from_scalar.py
|   |   |       |   |   |   |   |-- ✅ test_construct_ndarray.py
|   |   |       |   |   |   |   |-- ✅ test_construct_object_arr.py
|   |   |       |   |   |   |   |-- ✅ test_dict_compat.py
|   |   |       |   |   |   |   |-- ✅ test_downcast.py
|   |   |       |   |   |   |   |-- ✅ test_find_common_type.py
|   |   |       |   |   |   |   |-- ✅ test_infer_datetimelike.py
|   |   |       |   |   |   |   |-- ✅ test_infer_dtype.py
|   |   |       |   |   |   |   |-- ✅ test_maybe_box_native.py
|   |   |       |   |   |   |   \-- ✅ test_promote.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_concat.py
|   |   |       |   |   |   |-- ✅ test_dtypes.py
|   |   |       |   |   |   |-- ✅ test_generic.py
|   |   |       |   |   |   |-- ✅ test_inference.py
|   |   |       |   |   |   \-- ✅ test_missing.py
|   |   |       |   |   |-- ✅ extension/
|   |   |       |   |   |   |-- ✅ array_with_attr/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ array.py
|   |   |       |   |   |   |   \-- ✅ test_array_with_attr.py
|   |   |       |   |   |   |-- ✅ base/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ accumulate.py
|   |   |       |   |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |   |-- ✅ casting.py
|   |   |       |   |   |   |   |-- ✅ constructors.py
|   |   |       |   |   |   |   |-- ✅ dim2.py
|   |   |       |   |   |   |   |-- ✅ dtype.py
|   |   |       |   |   |   |   |-- ✅ getitem.py
|   |   |       |   |   |   |   |-- ✅ groupby.py
|   |   |       |   |   |   |   |-- ✅ index.py
|   |   |       |   |   |   |   |-- ✅ interface.py
|   |   |       |   |   |   |   |-- ✅ io.py
|   |   |       |   |   |   |   |-- ✅ methods.py
|   |   |       |   |   |   |   |-- ✅ missing.py
|   |   |       |   |   |   |   |-- ✅ ops.py
|   |   |       |   |   |   |   |-- ✅ printing.py
|   |   |       |   |   |   |   |-- ✅ reduce.py
|   |   |       |   |   |   |   |-- ✅ reshaping.py
|   |   |       |   |   |   |   \-- ✅ setitem.py
|   |   |       |   |   |   |-- ✅ date/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ array.py
|   |   |       |   |   |   |-- ✅ decimal/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ array.py
|   |   |       |   |   |   |   \-- ✅ test_decimal.py
|   |   |       |   |   |   |-- ✅ json/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ array.py
|   |   |       |   |   |   |   \-- ✅ test_json.py
|   |   |       |   |   |   |-- ✅ list/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ array.py
|   |   |       |   |   |   |   \-- ✅ test_list.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_arrow.py
|   |   |       |   |   |   |-- ✅ test_categorical.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_datetime.py
|   |   |       |   |   |   |-- ✅ test_extension.py
|   |   |       |   |   |   |-- ✅ test_interval.py
|   |   |       |   |   |   |-- ✅ test_masked.py
|   |   |       |   |   |   |-- ✅ test_numpy.py
|   |   |       |   |   |   |-- ✅ test_period.py
|   |   |       |   |   |   |-- ✅ test_sparse.py
|   |   |       |   |   |   \-- ✅ test_string.py
|   |   |       |   |   |-- ✅ frame/
|   |   |       |   |   |   |-- ✅ constructors/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_from_dict.py
|   |   |       |   |   |   |   \-- ✅ test_from_records.py
|   |   |       |   |   |   |-- ✅ indexing/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_coercion.py
|   |   |       |   |   |   |   |-- ✅ test_delitem.py
|   |   |       |   |   |   |   |-- ✅ test_get.py
|   |   |       |   |   |   |   |-- ✅ test_get_value.py
|   |   |       |   |   |   |   |-- ✅ test_getitem.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_insert.py
|   |   |       |   |   |   |   |-- ✅ test_mask.py
|   |   |       |   |   |   |   |-- ✅ test_set_value.py
|   |   |       |   |   |   |   |-- ✅ test_setitem.py
|   |   |       |   |   |   |   |-- ✅ test_take.py
|   |   |       |   |   |   |   |-- ✅ test_where.py
|   |   |       |   |   |   |   \-- ✅ test_xs.py
|   |   |       |   |   |   |-- ✅ methods/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_add_prefix_suffix.py
|   |   |       |   |   |   |   |-- ✅ test_align.py
|   |   |       |   |   |   |   |-- ✅ test_asfreq.py
|   |   |       |   |   |   |   |-- ✅ test_asof.py
|   |   |       |   |   |   |   |-- ✅ test_assign.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_at_time.py
|   |   |       |   |   |   |   |-- ✅ test_between_time.py
|   |   |       |   |   |   |   |-- ✅ test_clip.py
|   |   |       |   |   |   |   |-- ✅ test_combine.py
|   |   |       |   |   |   |   |-- ✅ test_combine_first.py
|   |   |       |   |   |   |   |-- ✅ test_compare.py
|   |   |       |   |   |   |   |-- ✅ test_convert_dtypes.py
|   |   |       |   |   |   |   |-- ✅ test_copy.py
|   |   |       |   |   |   |   |-- ✅ test_count.py
|   |   |       |   |   |   |   |-- ✅ test_cov_corr.py
|   |   |       |   |   |   |   |-- ✅ test_describe.py
|   |   |       |   |   |   |   |-- ✅ test_diff.py
|   |   |       |   |   |   |   |-- ✅ test_dot.py
|   |   |       |   |   |   |   |-- ✅ test_drop.py
|   |   |       |   |   |   |   |-- ✅ test_drop_duplicates.py
|   |   |       |   |   |   |   |-- ✅ test_droplevel.py
|   |   |       |   |   |   |   |-- ✅ test_dropna.py
|   |   |       |   |   |   |   |-- ✅ test_dtypes.py
|   |   |       |   |   |   |   |-- ✅ test_duplicated.py
|   |   |       |   |   |   |   |-- ✅ test_equals.py
|   |   |       |   |   |   |   |-- ✅ test_explode.py
|   |   |       |   |   |   |   |-- ✅ test_fillna.py
|   |   |       |   |   |   |   |-- ✅ test_filter.py
|   |   |       |   |   |   |   |-- ✅ test_first_and_last.py
|   |   |       |   |   |   |   |-- ✅ test_first_valid_index.py
|   |   |       |   |   |   |   |-- ✅ test_get_numeric_data.py
|   |   |       |   |   |   |   |-- ✅ test_head_tail.py
|   |   |       |   |   |   |   |-- ✅ test_infer_objects.py
|   |   |       |   |   |   |   |-- ✅ test_info.py
|   |   |       |   |   |   |   |-- ✅ test_interpolate.py
|   |   |       |   |   |   |   |-- ✅ test_is_homogeneous_dtype.py
|   |   |       |   |   |   |   |-- ✅ test_isetitem.py
|   |   |       |   |   |   |   |-- ✅ test_isin.py
|   |   |       |   |   |   |   |-- ✅ test_iterrows.py
|   |   |       |   |   |   |   |-- ✅ test_join.py
|   |   |       |   |   |   |   |-- ✅ test_map.py
|   |   |       |   |   |   |   |-- ✅ test_matmul.py
|   |   |       |   |   |   |   |-- ✅ test_nlargest.py
|   |   |       |   |   |   |   |-- ✅ test_pct_change.py
|   |   |       |   |   |   |   |-- ✅ test_pipe.py
|   |   |       |   |   |   |   |-- ✅ test_pop.py
|   |   |       |   |   |   |   |-- ✅ test_quantile.py
|   |   |       |   |   |   |   |-- ✅ test_rank.py
|   |   |       |   |   |   |   |-- ✅ test_reindex.py
|   |   |       |   |   |   |   |-- ✅ test_reindex_like.py
|   |   |       |   |   |   |   |-- ✅ test_rename.py
|   |   |       |   |   |   |   |-- ✅ test_rename_axis.py
|   |   |       |   |   |   |   |-- ✅ test_reorder_levels.py
|   |   |       |   |   |   |   |-- ✅ test_replace.py
|   |   |       |   |   |   |   |-- ✅ test_reset_index.py
|   |   |       |   |   |   |   |-- ✅ test_round.py
|   |   |       |   |   |   |   |-- ✅ test_sample.py
|   |   |       |   |   |   |   |-- ✅ test_select_dtypes.py
|   |   |       |   |   |   |   |-- ✅ test_set_axis.py
|   |   |       |   |   |   |   |-- ✅ test_set_index.py
|   |   |       |   |   |   |   |-- ✅ test_shift.py
|   |   |       |   |   |   |   |-- ✅ test_size.py
|   |   |       |   |   |   |   |-- ✅ test_sort_index.py
|   |   |       |   |   |   |   |-- ✅ test_sort_values.py
|   |   |       |   |   |   |   |-- ✅ test_swapaxes.py
|   |   |       |   |   |   |   |-- ✅ test_swaplevel.py
|   |   |       |   |   |   |   |-- ✅ test_to_csv.py
|   |   |       |   |   |   |   |-- ✅ test_to_dict.py
|   |   |       |   |   |   |   |-- ✅ test_to_dict_of_blocks.py
|   |   |       |   |   |   |   |-- ✅ test_to_numpy.py
|   |   |       |   |   |   |   |-- ✅ test_to_period.py
|   |   |       |   |   |   |   |-- ✅ test_to_records.py
|   |   |       |   |   |   |   |-- ✅ test_to_timestamp.py
|   |   |       |   |   |   |   |-- ✅ test_transpose.py
|   |   |       |   |   |   |   |-- ✅ test_truncate.py
|   |   |       |   |   |   |   |-- ✅ test_tz_convert.py
|   |   |       |   |   |   |   |-- ✅ test_tz_localize.py
|   |   |       |   |   |   |   |-- ✅ test_update.py
|   |   |       |   |   |   |   |-- ✅ test_value_counts.py
|   |   |       |   |   |   |   \-- ✅ test_values.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_alter_axes.py
|   |   |       |   |   |   |-- ✅ test_api.py
|   |   |       |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |-- ✅ test_arrow_interface.py
|   |   |       |   |   |   |-- ✅ test_block_internals.py
|   |   |       |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |-- ✅ test_cumulative.py
|   |   |       |   |   |   |-- ✅ test_iteration.py
|   |   |       |   |   |   |-- ✅ test_logical_ops.py
|   |   |       |   |   |   |-- ✅ test_nonunique_indexes.py
|   |   |       |   |   |   |-- ✅ test_npfuncs.py
|   |   |       |   |   |   |-- ✅ test_query_eval.py
|   |   |       |   |   |   |-- ✅ test_reductions.py
|   |   |       |   |   |   |-- ✅ test_repr.py
|   |   |       |   |   |   |-- ✅ test_stack_unstack.py
|   |   |       |   |   |   |-- ✅ test_subclass.py
|   |   |       |   |   |   |-- ✅ test_ufunc.py
|   |   |       |   |   |   |-- ✅ test_unary.py
|   |   |       |   |   |   \-- ✅ test_validate.py
|   |   |       |   |   |-- ✅ generic/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_duplicate_labels.py
|   |   |       |   |   |   |-- ✅ test_finalize.py
|   |   |       |   |   |   |-- ✅ test_frame.py
|   |   |       |   |   |   |-- ✅ test_generic.py
|   |   |       |   |   |   |-- ✅ test_label_or_level_utils.py
|   |   |       |   |   |   |-- ✅ test_series.py
|   |   |       |   |   |   \-- ✅ test_to_xarray.py
|   |   |       |   |   |-- ✅ groupby/
|   |   |       |   |   |   |-- ✅ aggregate/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_aggregate.py
|   |   |       |   |   |   |   |-- ✅ test_cython.py
|   |   |       |   |   |   |   |-- ✅ test_numba.py
|   |   |       |   |   |   |   \-- ✅ test_other.py
|   |   |       |   |   |   |-- ✅ methods/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_corrwith.py
|   |   |       |   |   |   |   |-- ✅ test_describe.py
|   |   |       |   |   |   |   |-- ✅ test_groupby_shift_diff.py
|   |   |       |   |   |   |   |-- ✅ test_is_monotonic.py
|   |   |       |   |   |   |   |-- ✅ test_nlargest_nsmallest.py
|   |   |       |   |   |   |   |-- ✅ test_nth.py
|   |   |       |   |   |   |   |-- ✅ test_quantile.py
|   |   |       |   |   |   |   |-- ✅ test_rank.py
|   |   |       |   |   |   |   |-- ✅ test_sample.py
|   |   |       |   |   |   |   |-- ✅ test_size.py
|   |   |       |   |   |   |   |-- ✅ test_skew.py
|   |   |       |   |   |   |   \-- ✅ test_value_counts.py
|   |   |       |   |   |   |-- ✅ transform/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_numba.py
|   |   |       |   |   |   |   \-- ✅ test_transform.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_all_methods.py
|   |   |       |   |   |   |-- ✅ test_api.py
|   |   |       |   |   |   |-- ✅ test_apply.py
|   |   |       |   |   |   |-- ✅ test_apply_mutate.py
|   |   |       |   |   |   |-- ✅ test_bin_groupby.py
|   |   |       |   |   |   |-- ✅ test_categorical.py
|   |   |       |   |   |   |-- ✅ test_counting.py
|   |   |       |   |   |   |-- ✅ test_cumulative.py
|   |   |       |   |   |   |-- ✅ test_filters.py
|   |   |       |   |   |   |-- ✅ test_groupby.py
|   |   |       |   |   |   |-- ✅ test_groupby_dropna.py
|   |   |       |   |   |   |-- ✅ test_groupby_subclass.py
|   |   |       |   |   |   |-- ✅ test_grouping.py
|   |   |       |   |   |   |-- ✅ test_index_as_string.py
|   |   |       |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |-- ✅ test_libgroupby.py
|   |   |       |   |   |   |-- ✅ test_missing.py
|   |   |       |   |   |   |-- ✅ test_numba.py
|   |   |       |   |   |   |-- ✅ test_numeric_only.py
|   |   |       |   |   |   |-- ✅ test_pipe.py
|   |   |       |   |   |   |-- ✅ test_raises.py
|   |   |       |   |   |   |-- ✅ test_reductions.py
|   |   |       |   |   |   \-- ✅ test_timegrouper.py
|   |   |       |   |   |-- ✅ indexes/
|   |   |       |   |   |   |-- ✅ base_class/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_pickle.py
|   |   |       |   |   |   |   |-- ✅ test_reshape.py
|   |   |       |   |   |   |   |-- ✅ test_setops.py
|   |   |       |   |   |   |   \-- ✅ test_where.py
|   |   |       |   |   |   |-- ✅ categorical/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_append.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_category.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_equals.py
|   |   |       |   |   |   |   |-- ✅ test_fillna.py
|   |   |       |   |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_map.py
|   |   |       |   |   |   |   |-- ✅ test_reindex.py
|   |   |       |   |   |   |   \-- ✅ test_setops.py
|   |   |       |   |   |   |-- ✅ datetimelike_/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_drop_duplicates.py
|   |   |       |   |   |   |   |-- ✅ test_equals.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_is_monotonic.py
|   |   |       |   |   |   |   |-- ✅ test_nat.py
|   |   |       |   |   |   |   |-- ✅ test_sort_values.py
|   |   |       |   |   |   |   \-- ✅ test_value_counts.py
|   |   |       |   |   |   |-- ✅ datetimes/
|   |   |       |   |   |   |   |-- ✅ methods/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ test_asof.py
|   |   |       |   |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |   |-- ✅ test_delete.py
|   |   |       |   |   |   |   |   |-- ✅ test_factorize.py
|   |   |       |   |   |   |   |   |-- ✅ test_fillna.py
|   |   |       |   |   |   |   |   |-- ✅ test_insert.py
|   |   |       |   |   |   |   |   |-- ✅ test_isocalendar.py
|   |   |       |   |   |   |   |   |-- ✅ test_map.py
|   |   |       |   |   |   |   |   |-- ✅ test_normalize.py
|   |   |       |   |   |   |   |   |-- ✅ test_repeat.py
|   |   |       |   |   |   |   |   |-- ✅ test_resolution.py
|   |   |       |   |   |   |   |   |-- ✅ test_round.py
|   |   |       |   |   |   |   |   |-- ✅ test_shift.py
|   |   |       |   |   |   |   |   |-- ✅ test_snap.py
|   |   |       |   |   |   |   |   |-- ✅ test_to_frame.py
|   |   |       |   |   |   |   |   |-- ✅ test_to_julian_date.py
|   |   |       |   |   |   |   |   |-- ✅ test_to_period.py
|   |   |       |   |   |   |   |   |-- ✅ test_to_pydatetime.py
|   |   |       |   |   |   |   |   |-- ✅ test_to_series.py
|   |   |       |   |   |   |   |   |-- ✅ test_tz_convert.py
|   |   |       |   |   |   |   |   |-- ✅ test_tz_localize.py
|   |   |       |   |   |   |   |   \-- ✅ test_unique.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_date_range.py
|   |   |       |   |   |   |   |-- ✅ test_datetime.py
|   |   |       |   |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |   |-- ✅ test_freq_attr.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_iter.py
|   |   |       |   |   |   |   |-- ✅ test_join.py
|   |   |       |   |   |   |   |-- ✅ test_npfuncs.py
|   |   |       |   |   |   |   |-- ✅ test_ops.py
|   |   |       |   |   |   |   |-- ✅ test_partial_slicing.py
|   |   |       |   |   |   |   |-- ✅ test_pickle.py
|   |   |       |   |   |   |   |-- ✅ test_reindex.py
|   |   |       |   |   |   |   |-- ✅ test_scalar_compat.py
|   |   |       |   |   |   |   |-- ✅ test_setops.py
|   |   |       |   |   |   |   \-- ✅ test_timezones.py
|   |   |       |   |   |   |-- ✅ interval/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_equals.py
|   |   |       |   |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_interval.py
|   |   |       |   |   |   |   |-- ✅ test_interval_range.py
|   |   |       |   |   |   |   |-- ✅ test_interval_tree.py
|   |   |       |   |   |   |   |-- ✅ test_join.py
|   |   |       |   |   |   |   |-- ✅ test_pickle.py
|   |   |       |   |   |   |   \-- ✅ test_setops.py
|   |   |       |   |   |   |-- ✅ multi/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |   |-- ✅ test_analytics.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_compat.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_conversion.py
|   |   |       |   |   |   |   |-- ✅ test_copy.py
|   |   |       |   |   |   |   |-- ✅ test_drop.py
|   |   |       |   |   |   |   |-- ✅ test_duplicates.py
|   |   |       |   |   |   |   |-- ✅ test_equivalence.py
|   |   |       |   |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |   |-- ✅ test_get_level_values.py
|   |   |       |   |   |   |   |-- ✅ test_get_set.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_integrity.py
|   |   |       |   |   |   |   |-- ✅ test_isin.py
|   |   |       |   |   |   |   |-- ✅ test_join.py
|   |   |       |   |   |   |   |-- ✅ test_lexsort.py
|   |   |       |   |   |   |   |-- ✅ test_missing.py
|   |   |       |   |   |   |   |-- ✅ test_monotonic.py
|   |   |       |   |   |   |   |-- ✅ test_names.py
|   |   |       |   |   |   |   |-- ✅ test_partial_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_pickle.py
|   |   |       |   |   |   |   |-- ✅ test_reindex.py
|   |   |       |   |   |   |   |-- ✅ test_reshape.py
|   |   |       |   |   |   |   |-- ✅ test_setops.py
|   |   |       |   |   |   |   |-- ✅ test_sorting.py
|   |   |       |   |   |   |   \-- ✅ test_take.py
|   |   |       |   |   |   |-- ✅ numeric/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_join.py
|   |   |       |   |   |   |   |-- ✅ test_numeric.py
|   |   |       |   |   |   |   \-- ✅ test_setops.py
|   |   |       |   |   |   |-- ✅ object/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   \-- ✅ test_indexing.py
|   |   |       |   |   |   |-- ✅ period/
|   |   |       |   |   |   |   |-- ✅ methods/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ test_asfreq.py
|   |   |       |   |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |   |-- ✅ test_factorize.py
|   |   |       |   |   |   |   |   |-- ✅ test_fillna.py
|   |   |       |   |   |   |   |   |-- ✅ test_insert.py
|   |   |       |   |   |   |   |   |-- ✅ test_is_full.py
|   |   |       |   |   |   |   |   |-- ✅ test_repeat.py
|   |   |       |   |   |   |   |   |-- ✅ test_shift.py
|   |   |       |   |   |   |   |   \-- ✅ test_to_timestamp.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |   |-- ✅ test_freq_attr.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_join.py
|   |   |       |   |   |   |   |-- ✅ test_monotonic.py
|   |   |       |   |   |   |   |-- ✅ test_partial_slicing.py
|   |   |       |   |   |   |   |-- ✅ test_period.py
|   |   |       |   |   |   |   |-- ✅ test_period_range.py
|   |   |       |   |   |   |   |-- ✅ test_pickle.py
|   |   |       |   |   |   |   |-- ✅ test_resolution.py
|   |   |       |   |   |   |   |-- ✅ test_scalar_compat.py
|   |   |       |   |   |   |   |-- ✅ test_searchsorted.py
|   |   |       |   |   |   |   |-- ✅ test_setops.py
|   |   |       |   |   |   |   \-- ✅ test_tools.py
|   |   |       |   |   |   |-- ✅ ranges/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_join.py
|   |   |       |   |   |   |   |-- ✅ test_range.py
|   |   |       |   |   |   |   \-- ✅ test_setops.py
|   |   |       |   |   |   |-- ✅ string/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   \-- ✅ test_indexing.py
|   |   |       |   |   |   |-- ✅ timedeltas/
|   |   |       |   |   |   |   |-- ✅ methods/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |   |-- ✅ test_factorize.py
|   |   |       |   |   |   |   |   |-- ✅ test_fillna.py
|   |   |       |   |   |   |   |   |-- ✅ test_insert.py
|   |   |       |   |   |   |   |   |-- ✅ test_repeat.py
|   |   |       |   |   |   |   |   \-- ✅ test_shift.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_delete.py
|   |   |       |   |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |   |-- ✅ test_freq_attr.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_join.py
|   |   |       |   |   |   |   |-- ✅ test_ops.py
|   |   |       |   |   |   |   |-- ✅ test_pickle.py
|   |   |       |   |   |   |   |-- ✅ test_scalar_compat.py
|   |   |       |   |   |   |   |-- ✅ test_searchsorted.py
|   |   |       |   |   |   |   |-- ✅ test_setops.py
|   |   |       |   |   |   |   |-- ✅ test_timedelta.py
|   |   |       |   |   |   |   \-- ✅ test_timedelta_range.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_any_index.py
|   |   |       |   |   |   |-- ✅ test_base.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_datetimelike.py
|   |   |       |   |   |   |-- ✅ test_engines.py
|   |   |       |   |   |   |-- ✅ test_frozen.py
|   |   |       |   |   |   |-- ✅ test_index_new.py
|   |   |       |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |-- ✅ test_numpy_compat.py
|   |   |       |   |   |   |-- ✅ test_old_base.py
|   |   |       |   |   |   |-- ✅ test_setops.py
|   |   |       |   |   |   \-- ✅ test_subclass.py
|   |   |       |   |   |-- ✅ indexing/
|   |   |       |   |   |   |-- ✅ interval/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_interval.py
|   |   |       |   |   |   |   \-- ✅ test_interval_new.py
|   |   |       |   |   |   |-- ✅ multiindex/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_chaining_and_caching.py
|   |   |       |   |   |   |   |-- ✅ test_datetime.py
|   |   |       |   |   |   |   |-- ✅ test_getitem.py
|   |   |       |   |   |   |   |-- ✅ test_iloc.py
|   |   |       |   |   |   |   |-- ✅ test_indexing_slow.py
|   |   |       |   |   |   |   |-- ✅ test_loc.py
|   |   |       |   |   |   |   |-- ✅ test_multiindex.py
|   |   |       |   |   |   |   |-- ✅ test_partial.py
|   |   |       |   |   |   |   |-- ✅ test_setitem.py
|   |   |       |   |   |   |   |-- ✅ test_slice.py
|   |   |       |   |   |   |   \-- ✅ test_sorted.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_at.py
|   |   |       |   |   |   |-- ✅ test_categorical.py
|   |   |       |   |   |   |-- ✅ test_chaining_and_caching.py
|   |   |       |   |   |   |-- ✅ test_check_indexer.py
|   |   |       |   |   |   |-- ✅ test_coercion.py
|   |   |       |   |   |   |-- ✅ test_datetime.py
|   |   |       |   |   |   |-- ✅ test_floats.py
|   |   |       |   |   |   |-- ✅ test_iat.py
|   |   |       |   |   |   |-- ✅ test_iloc.py
|   |   |       |   |   |   |-- ✅ test_indexers.py
|   |   |       |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |-- ✅ test_loc.py
|   |   |       |   |   |   |-- ✅ test_na_indexing.py
|   |   |       |   |   |   |-- ✅ test_partial.py
|   |   |       |   |   |   \-- ✅ test_scalar.py
|   |   |       |   |   |-- ✅ interchange/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_impl.py
|   |   |       |   |   |   |-- ✅ test_spec_conformance.py
|   |   |       |   |   |   \-- ✅ test_utils.py
|   |   |       |   |   |-- ✅ internals/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_api.py
|   |   |       |   |   |   |-- ✅ test_internals.py
|   |   |       |   |   |   \-- ✅ test_managers.py
|   |   |       |   |   |-- ✅ io/
|   |   |       |   |   |   |-- ✅ excel/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_odf.py
|   |   |       |   |   |   |   |-- ✅ test_odswriter.py
|   |   |       |   |   |   |   |-- ✅ test_openpyxl.py
|   |   |       |   |   |   |   |-- ✅ test_readers.py
|   |   |       |   |   |   |   |-- ✅ test_style.py
|   |   |       |   |   |   |   |-- ✅ test_writers.py
|   |   |       |   |   |   |   |-- ✅ test_xlrd.py
|   |   |       |   |   |   |   \-- ✅ test_xlsxwriter.py
|   |   |       |   |   |   |-- ✅ formats/
|   |   |       |   |   |   |   |-- ✅ style/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ test_bar.py
|   |   |       |   |   |   |   |   |-- ✅ test_exceptions.py
|   |   |       |   |   |   |   |   |-- ✅ test_format.py
|   |   |       |   |   |   |   |   |-- ✅ test_highlight.py
|   |   |       |   |   |   |   |   |-- ✅ test_html.py
|   |   |       |   |   |   |   |   |-- ✅ test_matplotlib.py
|   |   |       |   |   |   |   |   |-- ✅ test_non_unique.py
|   |   |       |   |   |   |   |   |-- ✅ test_style.py
|   |   |       |   |   |   |   |   |-- ✅ test_to_latex.py
|   |   |       |   |   |   |   |   |-- ✅ test_to_string.py
|   |   |       |   |   |   |   |   \-- ✅ test_tooltip.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_console.py
|   |   |       |   |   |   |   |-- ✅ test_css.py
|   |   |       |   |   |   |   |-- ✅ test_eng_formatting.py
|   |   |       |   |   |   |   |-- ✅ test_format.py
|   |   |       |   |   |   |   |-- ✅ test_ipython_compat.py
|   |   |       |   |   |   |   |-- ✅ test_printing.py
|   |   |       |   |   |   |   |-- ✅ test_to_csv.py
|   |   |       |   |   |   |   |-- ✅ test_to_excel.py
|   |   |       |   |   |   |   |-- ✅ test_to_html.py
|   |   |       |   |   |   |   |-- ✅ test_to_latex.py
|   |   |       |   |   |   |   |-- ✅ test_to_markdown.py
|   |   |       |   |   |   |   \-- ✅ test_to_string.py
|   |   |       |   |   |   |-- ✅ json/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |   |-- ✅ test_compression.py
|   |   |       |   |   |   |   |-- ✅ test_deprecated_kwargs.py
|   |   |       |   |   |   |   |-- ✅ test_json_table_schema.py
|   |   |       |   |   |   |   |-- ✅ test_json_table_schema_ext_dtype.py
|   |   |       |   |   |   |   |-- ✅ test_normalize.py
|   |   |       |   |   |   |   |-- ✅ test_pandas.py
|   |   |       |   |   |   |   |-- ✅ test_readlines.py
|   |   |       |   |   |   |   \-- ✅ test_ujson.py
|   |   |       |   |   |   |-- ✅ parser/
|   |   |       |   |   |   |   |-- ✅ common/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ test_chunksize.py
|   |   |       |   |   |   |   |   |-- ✅ test_common_basic.py
|   |   |       |   |   |   |   |   |-- ✅ test_data_list.py
|   |   |       |   |   |   |   |   |-- ✅ test_decimal.py
|   |   |       |   |   |   |   |   |-- ✅ test_file_buffer_url.py
|   |   |       |   |   |   |   |   |-- ✅ test_float.py
|   |   |       |   |   |   |   |   |-- ✅ test_index.py
|   |   |       |   |   |   |   |   |-- ✅ test_inf.py
|   |   |       |   |   |   |   |   |-- ✅ test_ints.py
|   |   |       |   |   |   |   |   |-- ✅ test_iterator.py
|   |   |       |   |   |   |   |   |-- ✅ test_read_errors.py
|   |   |       |   |   |   |   |   \-- ✅ test_verbose.py
|   |   |       |   |   |   |   |-- ✅ dtypes/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ test_categorical.py
|   |   |       |   |   |   |   |   |-- ✅ test_dtypes_basic.py
|   |   |       |   |   |   |   |   \-- ✅ test_empty.py
|   |   |       |   |   |   |   |-- ✅ usecols/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ test_parse_dates.py
|   |   |       |   |   |   |   |   |-- ✅ test_strings.py
|   |   |       |   |   |   |   |   \-- ✅ test_usecols_basic.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |   |-- ✅ test_c_parser_only.py
|   |   |       |   |   |   |   |-- ✅ test_comment.py
|   |   |       |   |   |   |   |-- ✅ test_compression.py
|   |   |       |   |   |   |   |-- ✅ test_concatenate_chunks.py
|   |   |       |   |   |   |   |-- ✅ test_converters.py
|   |   |       |   |   |   |   |-- ✅ test_dialect.py
|   |   |       |   |   |   |   |-- ✅ test_encoding.py
|   |   |       |   |   |   |   |-- ✅ test_header.py
|   |   |       |   |   |   |   |-- ✅ test_index_col.py
|   |   |       |   |   |   |   |-- ✅ test_mangle_dupes.py
|   |   |       |   |   |   |   |-- ✅ test_multi_thread.py
|   |   |       |   |   |   |   |-- ✅ test_na_values.py
|   |   |       |   |   |   |   |-- ✅ test_network.py
|   |   |       |   |   |   |   |-- ✅ test_parse_dates.py
|   |   |       |   |   |   |   |-- ✅ test_python_parser_only.py
|   |   |       |   |   |   |   |-- ✅ test_quoting.py
|   |   |       |   |   |   |   |-- ✅ test_read_fwf.py
|   |   |       |   |   |   |   |-- ✅ test_skiprows.py
|   |   |       |   |   |   |   |-- ✅ test_textreader.py
|   |   |       |   |   |   |   |-- ✅ test_unsupported.py
|   |   |       |   |   |   |   \-- ✅ test_upcast.py
|   |   |       |   |   |   |-- ✅ pytables/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |   |-- ✅ test_append.py
|   |   |       |   |   |   |   |-- ✅ test_categorical.py
|   |   |       |   |   |   |   |-- ✅ test_compat.py
|   |   |       |   |   |   |   |-- ✅ test_complex.py
|   |   |       |   |   |   |   |-- ✅ test_errors.py
|   |   |       |   |   |   |   |-- ✅ test_file_handling.py
|   |   |       |   |   |   |   |-- ✅ test_keys.py
|   |   |       |   |   |   |   |-- ✅ test_put.py
|   |   |       |   |   |   |   |-- ✅ test_pytables_missing.py
|   |   |       |   |   |   |   |-- ✅ test_read.py
|   |   |       |   |   |   |   |-- ✅ test_retain_attributes.py
|   |   |       |   |   |   |   |-- ✅ test_round_trip.py
|   |   |       |   |   |   |   |-- ✅ test_select.py
|   |   |       |   |   |   |   |-- ✅ test_store.py
|   |   |       |   |   |   |   |-- ✅ test_subclass.py
|   |   |       |   |   |   |   |-- ✅ test_time_series.py
|   |   |       |   |   |   |   \-- ✅ test_timezones.py
|   |   |       |   |   |   |-- ✅ sas/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_byteswap.py
|   |   |       |   |   |   |   |-- ✅ test_sas.py
|   |   |       |   |   |   |   |-- ✅ test_sas7bdat.py
|   |   |       |   |   |   |   \-- ✅ test_xport.py
|   |   |       |   |   |   |-- ✅ xml/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |   |-- ✅ test_to_xml.py
|   |   |       |   |   |   |   |-- ✅ test_xml.py
|   |   |       |   |   |   |   \-- ✅ test_xml_dtypes.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ generate_legacy_storage_files.py
|   |   |       |   |   |   |-- ✅ test_clipboard.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_compression.py
|   |   |       |   |   |   |-- ✅ test_feather.py
|   |   |       |   |   |   |-- ✅ test_fsspec.py
|   |   |       |   |   |   |-- ✅ test_gbq.py
|   |   |       |   |   |   |-- ✅ test_gcs.py
|   |   |       |   |   |   |-- ✅ test_html.py
|   |   |       |   |   |   |-- ✅ test_http_headers.py
|   |   |       |   |   |   |-- ✅ test_orc.py
|   |   |       |   |   |   |-- ✅ test_parquet.py
|   |   |       |   |   |   |-- ✅ test_pickle.py
|   |   |       |   |   |   |-- ✅ test_s3.py
|   |   |       |   |   |   |-- ✅ test_spss.py
|   |   |       |   |   |   |-- ✅ test_sql.py
|   |   |       |   |   |   \-- ✅ test_stata.py
|   |   |       |   |   |-- ✅ libs/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_hashtable.py
|   |   |       |   |   |   |-- ✅ test_join.py
|   |   |       |   |   |   |-- ✅ test_lib.py
|   |   |       |   |   |   \-- ✅ test_libalgos.py
|   |   |       |   |   |-- ✅ plotting/
|   |   |       |   |   |   |-- ✅ frame/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_frame.py
|   |   |       |   |   |   |   |-- ✅ test_frame_color.py
|   |   |       |   |   |   |   |-- ✅ test_frame_groupby.py
|   |   |       |   |   |   |   |-- ✅ test_frame_legend.py
|   |   |       |   |   |   |   |-- ✅ test_frame_subplots.py
|   |   |       |   |   |   |   \-- ✅ test_hist_box_by.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_backend.py
|   |   |       |   |   |   |-- ✅ test_boxplot_method.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_converter.py
|   |   |       |   |   |   |-- ✅ test_datetimelike.py
|   |   |       |   |   |   |-- ✅ test_groupby.py
|   |   |       |   |   |   |-- ✅ test_hist_method.py
|   |   |       |   |   |   |-- ✅ test_misc.py
|   |   |       |   |   |   |-- ✅ test_series.py
|   |   |       |   |   |   \-- ✅ test_style.py
|   |   |       |   |   |-- ✅ reductions/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_reductions.py
|   |   |       |   |   |   \-- ✅ test_stat_reductions.py
|   |   |       |   |   |-- ✅ resample/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_base.py
|   |   |       |   |   |   |-- ✅ test_datetime_index.py
|   |   |       |   |   |   |-- ✅ test_period_index.py
|   |   |       |   |   |   |-- ✅ test_resample_api.py
|   |   |       |   |   |   |-- ✅ test_resampler_grouper.py
|   |   |       |   |   |   |-- ✅ test_time_grouper.py
|   |   |       |   |   |   \-- ✅ test_timedelta.py
|   |   |       |   |   |-- ✅ reshape/
|   |   |       |   |   |   |-- ✅ concat/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |   |-- ✅ test_append.py
|   |   |       |   |   |   |   |-- ✅ test_append_common.py
|   |   |       |   |   |   |   |-- ✅ test_categorical.py
|   |   |       |   |   |   |   |-- ✅ test_concat.py
|   |   |       |   |   |   |   |-- ✅ test_dataframe.py
|   |   |       |   |   |   |   |-- ✅ test_datetimes.py
|   |   |       |   |   |   |   |-- ✅ test_empty.py
|   |   |       |   |   |   |   |-- ✅ test_index.py
|   |   |       |   |   |   |   |-- ✅ test_invalid.py
|   |   |       |   |   |   |   |-- ✅ test_series.py
|   |   |       |   |   |   |   \-- ✅ test_sort.py
|   |   |       |   |   |   |-- ✅ merge/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_join.py
|   |   |       |   |   |   |   |-- ✅ test_merge.py
|   |   |       |   |   |   |   |-- ✅ test_merge_asof.py
|   |   |       |   |   |   |   |-- ✅ test_merge_cross.py
|   |   |       |   |   |   |   |-- ✅ test_merge_index_as_string.py
|   |   |       |   |   |   |   |-- ✅ test_merge_ordered.py
|   |   |       |   |   |   |   \-- ✅ test_multi.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_crosstab.py
|   |   |       |   |   |   |-- ✅ test_cut.py
|   |   |       |   |   |   |-- ✅ test_from_dummies.py
|   |   |       |   |   |   |-- ✅ test_get_dummies.py
|   |   |       |   |   |   |-- ✅ test_melt.py
|   |   |       |   |   |   |-- ✅ test_pivot.py
|   |   |       |   |   |   |-- ✅ test_pivot_multilevel.py
|   |   |       |   |   |   |-- ✅ test_qcut.py
|   |   |       |   |   |   |-- ✅ test_union_categoricals.py
|   |   |       |   |   |   \-- ✅ test_util.py
|   |   |       |   |   |-- ✅ scalar/
|   |   |       |   |   |   |-- ✅ interval/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_contains.py
|   |   |       |   |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |   |-- ✅ test_interval.py
|   |   |       |   |   |   |   \-- ✅ test_overlaps.py
|   |   |       |   |   |   |-- ✅ period/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |   |-- ✅ test_asfreq.py
|   |   |       |   |   |   |   \-- ✅ test_period.py
|   |   |       |   |   |   |-- ✅ timedelta/
|   |   |       |   |   |   |   |-- ✅ methods/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ test_as_unit.py
|   |   |       |   |   |   |   |   \-- ✅ test_round.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |   \-- ✅ test_timedelta.py
|   |   |       |   |   |   |-- ✅ timestamp/
|   |   |       |   |   |   |   |-- ✅ methods/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ test_as_unit.py
|   |   |       |   |   |   |   |   |-- ✅ test_normalize.py
|   |   |       |   |   |   |   |   |-- ✅ test_replace.py
|   |   |       |   |   |   |   |   |-- ✅ test_round.py
|   |   |       |   |   |   |   |   |-- ✅ test_timestamp_method.py
|   |   |       |   |   |   |   |   |-- ✅ test_to_julian_date.py
|   |   |       |   |   |   |   |   |-- ✅ test_to_pydatetime.py
|   |   |       |   |   |   |   |   |-- ✅ test_tz_convert.py
|   |   |       |   |   |   |   |   \-- ✅ test_tz_localize.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |   |-- ✅ test_comparisons.py
|   |   |       |   |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |   |-- ✅ test_timestamp.py
|   |   |       |   |   |   |   \-- ✅ test_timezones.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_na_scalar.py
|   |   |       |   |   |   \-- ✅ test_nat.py
|   |   |       |   |   |-- ✅ series/
|   |   |       |   |   |   |-- ✅ accessors/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_cat_accessor.py
|   |   |       |   |   |   |   |-- ✅ test_dt_accessor.py
|   |   |       |   |   |   |   |-- ✅ test_list_accessor.py
|   |   |       |   |   |   |   |-- ✅ test_sparse_accessor.py
|   |   |       |   |   |   |   |-- ✅ test_str_accessor.py
|   |   |       |   |   |   |   \-- ✅ test_struct_accessor.py
|   |   |       |   |   |   |-- ✅ indexing/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_datetime.py
|   |   |       |   |   |   |   |-- ✅ test_delitem.py
|   |   |       |   |   |   |   |-- ✅ test_get.py
|   |   |       |   |   |   |   |-- ✅ test_getitem.py
|   |   |       |   |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |   |-- ✅ test_mask.py
|   |   |       |   |   |   |   |-- ✅ test_set_value.py
|   |   |       |   |   |   |   |-- ✅ test_setitem.py
|   |   |       |   |   |   |   |-- ✅ test_take.py
|   |   |       |   |   |   |   |-- ✅ test_where.py
|   |   |       |   |   |   |   \-- ✅ test_xs.py
|   |   |       |   |   |   |-- ✅ methods/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_add_prefix_suffix.py
|   |   |       |   |   |   |   |-- ✅ test_align.py
|   |   |       |   |   |   |   |-- ✅ test_argsort.py
|   |   |       |   |   |   |   |-- ✅ test_asof.py
|   |   |       |   |   |   |   |-- ✅ test_astype.py
|   |   |       |   |   |   |   |-- ✅ test_autocorr.py
|   |   |       |   |   |   |   |-- ✅ test_between.py
|   |   |       |   |   |   |   |-- ✅ test_case_when.py
|   |   |       |   |   |   |   |-- ✅ test_clip.py
|   |   |       |   |   |   |   |-- ✅ test_combine.py
|   |   |       |   |   |   |   |-- ✅ test_combine_first.py
|   |   |       |   |   |   |   |-- ✅ test_compare.py
|   |   |       |   |   |   |   |-- ✅ test_convert_dtypes.py
|   |   |       |   |   |   |   |-- ✅ test_copy.py
|   |   |       |   |   |   |   |-- ✅ test_count.py
|   |   |       |   |   |   |   |-- ✅ test_cov_corr.py
|   |   |       |   |   |   |   |-- ✅ test_describe.py
|   |   |       |   |   |   |   |-- ✅ test_diff.py
|   |   |       |   |   |   |   |-- ✅ test_drop.py
|   |   |       |   |   |   |   |-- ✅ test_drop_duplicates.py
|   |   |       |   |   |   |   |-- ✅ test_dropna.py
|   |   |       |   |   |   |   |-- ✅ test_dtypes.py
|   |   |       |   |   |   |   |-- ✅ test_duplicated.py
|   |   |       |   |   |   |   |-- ✅ test_equals.py
|   |   |       |   |   |   |   |-- ✅ test_explode.py
|   |   |       |   |   |   |   |-- ✅ test_fillna.py
|   |   |       |   |   |   |   |-- ✅ test_get_numeric_data.py
|   |   |       |   |   |   |   |-- ✅ test_head_tail.py
|   |   |       |   |   |   |   |-- ✅ test_infer_objects.py
|   |   |       |   |   |   |   |-- ✅ test_info.py
|   |   |       |   |   |   |   |-- ✅ test_interpolate.py
|   |   |       |   |   |   |   |-- ✅ test_is_monotonic.py
|   |   |       |   |   |   |   |-- ✅ test_is_unique.py
|   |   |       |   |   |   |   |-- ✅ test_isin.py
|   |   |       |   |   |   |   |-- ✅ test_isna.py
|   |   |       |   |   |   |   |-- ✅ test_item.py
|   |   |       |   |   |   |   |-- ✅ test_map.py
|   |   |       |   |   |   |   |-- ✅ test_matmul.py
|   |   |       |   |   |   |   |-- ✅ test_nlargest.py
|   |   |       |   |   |   |   |-- ✅ test_nunique.py
|   |   |       |   |   |   |   |-- ✅ test_pct_change.py
|   |   |       |   |   |   |   |-- ✅ test_pop.py
|   |   |       |   |   |   |   |-- ✅ test_quantile.py
|   |   |       |   |   |   |   |-- ✅ test_rank.py
|   |   |       |   |   |   |   |-- ✅ test_reindex.py
|   |   |       |   |   |   |   |-- ✅ test_reindex_like.py
|   |   |       |   |   |   |   |-- ✅ test_rename.py
|   |   |       |   |   |   |   |-- ✅ test_rename_axis.py
|   |   |       |   |   |   |   |-- ✅ test_repeat.py
|   |   |       |   |   |   |   |-- ✅ test_replace.py
|   |   |       |   |   |   |   |-- ✅ test_reset_index.py
|   |   |       |   |   |   |   |-- ✅ test_round.py
|   |   |       |   |   |   |   |-- ✅ test_searchsorted.py
|   |   |       |   |   |   |   |-- ✅ test_set_name.py
|   |   |       |   |   |   |   |-- ✅ test_size.py
|   |   |       |   |   |   |   |-- ✅ test_sort_index.py
|   |   |       |   |   |   |   |-- ✅ test_sort_values.py
|   |   |       |   |   |   |   |-- ✅ test_to_csv.py
|   |   |       |   |   |   |   |-- ✅ test_to_dict.py
|   |   |       |   |   |   |   |-- ✅ test_to_frame.py
|   |   |       |   |   |   |   |-- ✅ test_to_numpy.py
|   |   |       |   |   |   |   |-- ✅ test_tolist.py
|   |   |       |   |   |   |   |-- ✅ test_truncate.py
|   |   |       |   |   |   |   |-- ✅ test_tz_localize.py
|   |   |       |   |   |   |   |-- ✅ test_unique.py
|   |   |       |   |   |   |   |-- ✅ test_unstack.py
|   |   |       |   |   |   |   |-- ✅ test_update.py
|   |   |       |   |   |   |   |-- ✅ test_value_counts.py
|   |   |       |   |   |   |   |-- ✅ test_values.py
|   |   |       |   |   |   |   \-- ✅ test_view.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_api.py
|   |   |       |   |   |   |-- ✅ test_arithmetic.py
|   |   |       |   |   |   |-- ✅ test_constructors.py
|   |   |       |   |   |   |-- ✅ test_cumulative.py
|   |   |       |   |   |   |-- ✅ test_formats.py
|   |   |       |   |   |   |-- ✅ test_iteration.py
|   |   |       |   |   |   |-- ✅ test_logical_ops.py
|   |   |       |   |   |   |-- ✅ test_missing.py
|   |   |       |   |   |   |-- ✅ test_npfuncs.py
|   |   |       |   |   |   |-- ✅ test_reductions.py
|   |   |       |   |   |   |-- ✅ test_subclass.py
|   |   |       |   |   |   |-- ✅ test_ufunc.py
|   |   |       |   |   |   |-- ✅ test_unary.py
|   |   |       |   |   |   \-- ✅ test_validate.py
|   |   |       |   |   |-- ✅ strings/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_api.py
|   |   |       |   |   |   |-- ✅ test_case_justify.py
|   |   |       |   |   |   |-- ✅ test_cat.py
|   |   |       |   |   |   |-- ✅ test_extract.py
|   |   |       |   |   |   |-- ✅ test_find_replace.py
|   |   |       |   |   |   |-- ✅ test_get_dummies.py
|   |   |       |   |   |   |-- ✅ test_split_partition.py
|   |   |       |   |   |   |-- ✅ test_string_array.py
|   |   |       |   |   |   \-- ✅ test_strings.py
|   |   |       |   |   |-- ✅ tools/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_to_datetime.py
|   |   |       |   |   |   |-- ✅ test_to_numeric.py
|   |   |       |   |   |   |-- ✅ test_to_time.py
|   |   |       |   |   |   \-- ✅ test_to_timedelta.py
|   |   |       |   |   |-- ✅ tseries/
|   |   |       |   |   |   |-- ✅ frequencies/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_freq_code.py
|   |   |       |   |   |   |   |-- ✅ test_frequencies.py
|   |   |       |   |   |   |   \-- ✅ test_inference.py
|   |   |       |   |   |   |-- ✅ holiday/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_calendar.py
|   |   |       |   |   |   |   |-- ✅ test_federal.py
|   |   |       |   |   |   |   |-- ✅ test_holiday.py
|   |   |       |   |   |   |   \-- ✅ test_observance.py
|   |   |       |   |   |   |-- ✅ offsets/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |   |-- ✅ test_business_day.py
|   |   |       |   |   |   |   |-- ✅ test_business_hour.py
|   |   |       |   |   |   |   |-- ✅ test_business_month.py
|   |   |       |   |   |   |   |-- ✅ test_business_quarter.py
|   |   |       |   |   |   |   |-- ✅ test_business_year.py
|   |   |       |   |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |   |-- ✅ test_custom_business_day.py
|   |   |       |   |   |   |   |-- ✅ test_custom_business_hour.py
|   |   |       |   |   |   |   |-- ✅ test_custom_business_month.py
|   |   |       |   |   |   |   |-- ✅ test_dst.py
|   |   |       |   |   |   |   |-- ✅ test_easter.py
|   |   |       |   |   |   |   |-- ✅ test_fiscal.py
|   |   |       |   |   |   |   |-- ✅ test_index.py
|   |   |       |   |   |   |   |-- ✅ test_month.py
|   |   |       |   |   |   |   |-- ✅ test_offsets.py
|   |   |       |   |   |   |   |-- ✅ test_offsets_properties.py
|   |   |       |   |   |   |   |-- ✅ test_quarter.py
|   |   |       |   |   |   |   |-- ✅ test_ticks.py
|   |   |       |   |   |   |   |-- ✅ test_week.py
|   |   |       |   |   |   |   \-- ✅ test_year.py
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ tslibs/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_api.py
|   |   |       |   |   |   |-- ✅ test_array_to_datetime.py
|   |   |       |   |   |   |-- ✅ test_ccalendar.py
|   |   |       |   |   |   |-- ✅ test_conversion.py
|   |   |       |   |   |   |-- ✅ test_fields.py
|   |   |       |   |   |   |-- ✅ test_libfrequencies.py
|   |   |       |   |   |   |-- ✅ test_liboffsets.py
|   |   |       |   |   |   |-- ✅ test_np_datetime.py
|   |   |       |   |   |   |-- ✅ test_npy_units.py
|   |   |       |   |   |   |-- ✅ test_parse_iso8601.py
|   |   |       |   |   |   |-- ✅ test_parsing.py
|   |   |       |   |   |   |-- ✅ test_period.py
|   |   |       |   |   |   |-- ✅ test_resolution.py
|   |   |       |   |   |   |-- ✅ test_strptime.py
|   |   |       |   |   |   |-- ✅ test_timedeltas.py
|   |   |       |   |   |   |-- ✅ test_timezones.py
|   |   |       |   |   |   |-- ✅ test_to_offset.py
|   |   |       |   |   |   \-- ✅ test_tzconversion.py
|   |   |       |   |   |-- ✅ util/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_assert_almost_equal.py
|   |   |       |   |   |   |-- ✅ test_assert_attr_equal.py
|   |   |       |   |   |   |-- ✅ test_assert_categorical_equal.py
|   |   |       |   |   |   |-- ✅ test_assert_extension_array_equal.py
|   |   |       |   |   |   |-- ✅ test_assert_frame_equal.py
|   |   |       |   |   |   |-- ✅ test_assert_index_equal.py
|   |   |       |   |   |   |-- ✅ test_assert_interval_array_equal.py
|   |   |       |   |   |   |-- ✅ test_assert_numpy_array_equal.py
|   |   |       |   |   |   |-- ✅ test_assert_produces_warning.py
|   |   |       |   |   |   |-- ✅ test_assert_series_equal.py
|   |   |       |   |   |   |-- ✅ test_deprecate.py
|   |   |       |   |   |   |-- ✅ test_deprecate_kwarg.py
|   |   |       |   |   |   |-- ✅ test_deprecate_nonkeyword_arguments.py
|   |   |       |   |   |   |-- ✅ test_doc.py
|   |   |       |   |   |   |-- ✅ test_hashing.py
|   |   |       |   |   |   |-- ✅ test_numba.py
|   |   |       |   |   |   |-- ✅ test_rewrite_warning.py
|   |   |       |   |   |   |-- ✅ test_shares_memory.py
|   |   |       |   |   |   |-- ✅ test_show_versions.py
|   |   |       |   |   |   |-- ✅ test_util.py
|   |   |       |   |   |   |-- ✅ test_validate_args.py
|   |   |       |   |   |   |-- ✅ test_validate_args_and_kwargs.py
|   |   |       |   |   |   |-- ✅ test_validate_inclusive.py
|   |   |       |   |   |   \-- ✅ test_validate_kwargs.py
|   |   |       |   |   |-- ✅ window/
|   |   |       |   |   |   |-- ✅ moments/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |   |-- ✅ test_moments_consistency_ewm.py
|   |   |       |   |   |   |   |-- ✅ test_moments_consistency_expanding.py
|   |   |       |   |   |   |   \-- ✅ test_moments_consistency_rolling.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ conftest.py
|   |   |       |   |   |   |-- ✅ test_api.py
|   |   |       |   |   |   |-- ✅ test_apply.py
|   |   |       |   |   |   |-- ✅ test_base_indexer.py
|   |   |       |   |   |   |-- ✅ test_cython_aggregations.py
|   |   |       |   |   |   |-- ✅ test_dtypes.py
|   |   |       |   |   |   |-- ✅ test_ewm.py
|   |   |       |   |   |   |-- ✅ test_expanding.py
|   |   |       |   |   |   |-- ✅ test_groupby.py
|   |   |       |   |   |   |-- ✅ test_numba.py
|   |   |       |   |   |   |-- ✅ test_online.py
|   |   |       |   |   |   |-- ✅ test_pairwise.py
|   |   |       |   |   |   |-- ✅ test_rolling.py
|   |   |       |   |   |   |-- ✅ test_rolling_functions.py
|   |   |       |   |   |   |-- ✅ test_rolling_quantile.py
|   |   |       |   |   |   |-- ✅ test_rolling_skew_kurt.py
|   |   |       |   |   |   |-- ✅ test_timeseries_window.py
|   |   |       |   |   |   \-- ✅ test_win_type.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ test_aggregation.py
|   |   |       |   |   |-- ✅ test_algos.py
|   |   |       |   |   |-- ✅ test_common.py
|   |   |       |   |   |-- ✅ test_downstream.py
|   |   |       |   |   |-- ✅ test_errors.py
|   |   |       |   |   |-- ✅ test_expressions.py
|   |   |       |   |   |-- ✅ test_flags.py
|   |   |       |   |   |-- ✅ test_multilevel.py
|   |   |       |   |   |-- ✅ test_nanops.py
|   |   |       |   |   |-- ✅ test_optional_dependency.py
|   |   |       |   |   |-- ✅ test_register_accessor.py
|   |   |       |   |   |-- ✅ test_sorting.py
|   |   |       |   |   \-- ✅ test_take.py
|   |   |       |   |-- ✅ tseries/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ api.py
|   |   |       |   |   |-- ✅ frequencies.py
|   |   |       |   |   |-- ✅ holiday.py
|   |   |       |   |   \-- ✅ offsets.py
|   |   |       |   |-- ✅ util/
|   |   |       |   |   |-- ✅ version/
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _decorators.py
|   |   |       |   |   |-- ✅ _doctools.py
|   |   |       |   |   |-- ✅ _exceptions.py
|   |   |       |   |   |-- ✅ _print_versions.py
|   |   |       |   |   |-- ✅ _test_decorators.py
|   |   |       |   |   |-- ✅ _tester.py
|   |   |       |   |   \-- ✅ _validators.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _typing.py
|   |   |       |   |-- ✅ _version.py
|   |   |       |   |-- ✅ _version_meson.py
|   |   |       |   |-- ✅ conftest.py
|   |   |       |   |-- ✅ pyproject.toml
|   |   |       |   \-- ✅ testing.py
|   |   |       |-- ✅ pandas-2.3.2.dist-info/
|   |   |       |   |-- ✅ DELVEWHEEL
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ pandas.libs/
|   |   |       |   \-- ✅ msvcp140-a4c2229bdc2a2a630acdc095b4d86008.dll
|   |   |       |-- ✅ PIL/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __main__.py
|   |   |       |   |-- ✅ _avif.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _avif.pyi
|   |   |       |   |-- ✅ _binary.py
|   |   |       |   |-- ✅ _deprecate.py
|   |   |       |   |-- ✅ _imaging.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _imaging.pyi
|   |   |       |   |-- ✅ _imagingcms.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _imagingcms.pyi
|   |   |       |   |-- ✅ _imagingft.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _imagingft.pyi
|   |   |       |   |-- ✅ _imagingmath.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _imagingmath.pyi
|   |   |       |   |-- ✅ _imagingmorph.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _imagingmorph.pyi
|   |   |       |   |-- ✅ _imagingtk.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _imagingtk.pyi
|   |   |       |   |-- ✅ _tkinter_finder.py
|   |   |       |   |-- ✅ _typing.py
|   |   |       |   |-- ✅ _util.py
|   |   |       |   |-- ✅ _version.py
|   |   |       |   |-- ✅ _webp.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _webp.pyi
|   |   |       |   |-- ✅ AvifImagePlugin.py
|   |   |       |   |-- ✅ BdfFontFile.py
|   |   |       |   |-- ✅ BlpImagePlugin.py
|   |   |       |   |-- ✅ BmpImagePlugin.py
|   |   |       |   |-- ✅ BufrStubImagePlugin.py
|   |   |       |   |-- ✅ ContainerIO.py
|   |   |       |   |-- ✅ CurImagePlugin.py
|   |   |       |   |-- ✅ DcxImagePlugin.py
|   |   |       |   |-- ✅ DdsImagePlugin.py
|   |   |       |   |-- ✅ EpsImagePlugin.py
|   |   |       |   |-- ✅ ExifTags.py
|   |   |       |   |-- ✅ features.py
|   |   |       |   |-- ✅ FitsImagePlugin.py
|   |   |       |   |-- ✅ FliImagePlugin.py
|   |   |       |   |-- ✅ FontFile.py
|   |   |       |   |-- ✅ FpxImagePlugin.py
|   |   |       |   |-- ✅ FtexImagePlugin.py
|   |   |       |   |-- ✅ GbrImagePlugin.py
|   |   |       |   |-- ✅ GdImageFile.py
|   |   |       |   |-- ✅ GifImagePlugin.py
|   |   |       |   |-- ✅ GimpGradientFile.py
|   |   |       |   |-- ✅ GimpPaletteFile.py
|   |   |       |   |-- ✅ GribStubImagePlugin.py
|   |   |       |   |-- ✅ Hdf5StubImagePlugin.py
|   |   |       |   |-- ✅ IcnsImagePlugin.py
|   |   |       |   |-- ✅ IcoImagePlugin.py
|   |   |       |   |-- ✅ Image.py
|   |   |       |   |-- ✅ ImageChops.py
|   |   |       |   |-- ✅ ImageCms.py
|   |   |       |   |-- ✅ ImageColor.py
|   |   |       |   |-- ✅ ImageDraw.py
|   |   |       |   |-- ✅ ImageDraw2.py
|   |   |       |   |-- ✅ ImageEnhance.py
|   |   |       |   |-- ✅ ImageFile.py
|   |   |       |   |-- ✅ ImageFilter.py
|   |   |       |   |-- ✅ ImageFont.py
|   |   |       |   |-- ✅ ImageGrab.py
|   |   |       |   |-- ✅ ImageMath.py
|   |   |       |   |-- ✅ ImageMode.py
|   |   |       |   |-- ✅ ImageMorph.py
|   |   |       |   |-- ✅ ImageOps.py
|   |   |       |   |-- ✅ ImagePalette.py
|   |   |       |   |-- ✅ ImagePath.py
|   |   |       |   |-- ✅ ImageQt.py
|   |   |       |   |-- ✅ ImageSequence.py
|   |   |       |   |-- ✅ ImageShow.py
|   |   |       |   |-- ✅ ImageStat.py
|   |   |       |   |-- ✅ ImageTk.py
|   |   |       |   |-- ✅ ImageTransform.py
|   |   |       |   |-- ✅ ImageWin.py
|   |   |       |   |-- ✅ ImImagePlugin.py
|   |   |       |   |-- ✅ ImtImagePlugin.py
|   |   |       |   |-- ✅ IptcImagePlugin.py
|   |   |       |   |-- ✅ Jpeg2KImagePlugin.py
|   |   |       |   |-- ✅ JpegImagePlugin.py
|   |   |       |   |-- ✅ JpegPresets.py
|   |   |       |   |-- ✅ McIdasImagePlugin.py
|   |   |       |   |-- ✅ MicImagePlugin.py
|   |   |       |   |-- ✅ MpegImagePlugin.py
|   |   |       |   |-- ✅ MpoImagePlugin.py
|   |   |       |   |-- ✅ MspImagePlugin.py
|   |   |       |   |-- ✅ PaletteFile.py
|   |   |       |   |-- ✅ PalmImagePlugin.py
|   |   |       |   |-- ✅ PcdImagePlugin.py
|   |   |       |   |-- ✅ PcfFontFile.py
|   |   |       |   |-- ✅ PcxImagePlugin.py
|   |   |       |   |-- ✅ PdfImagePlugin.py
|   |   |       |   |-- ✅ PdfParser.py
|   |   |       |   |-- ✅ PixarImagePlugin.py
|   |   |       |   |-- ✅ PngImagePlugin.py
|   |   |       |   |-- ✅ PpmImagePlugin.py
|   |   |       |   |-- ✅ PsdImagePlugin.py
|   |   |       |   |-- ✅ PSDraw.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ QoiImagePlugin.py
|   |   |       |   |-- ✅ report.py
|   |   |       |   |-- ✅ SgiImagePlugin.py
|   |   |       |   |-- ✅ SpiderImagePlugin.py
|   |   |       |   |-- ✅ SunImagePlugin.py
|   |   |       |   |-- ✅ TarIO.py
|   |   |       |   |-- ✅ TgaImagePlugin.py
|   |   |       |   |-- ✅ TiffImagePlugin.py
|   |   |       |   |-- ✅ TiffTags.py
|   |   |       |   |-- ✅ WalImageFile.py
|   |   |       |   |-- ✅ WebPImagePlugin.py
|   |   |       |   |-- ✅ WmfImagePlugin.py
|   |   |       |   |-- ✅ XbmImagePlugin.py
|   |   |       |   |-- ✅ XpmImagePlugin.py
|   |   |       |   \-- ✅ XVThumbImagePlugin.py
|   |   |       |-- ✅ pillow-11.3.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   |-- ✅ WHEEL
|   |   |       |   \-- ✅ zip-safe
|   |   |       |-- ✅ pip/
|   |   |       |   |-- ✅ _internal/
|   |   |       |   |   |-- ✅ cli/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ autocompletion.py
|   |   |       |   |   |   |-- ✅ base_command.py
|   |   |       |   |   |   |-- ✅ cmdoptions.py
|   |   |       |   |   |   |-- ✅ command_context.py
|   |   |       |   |   |   |-- ✅ index_command.py
|   |   |       |   |   |   |-- ✅ main.py
|   |   |       |   |   |   |-- ✅ main_parser.py
|   |   |       |   |   |   |-- ✅ parser.py
|   |   |       |   |   |   |-- ✅ progress_bars.py
|   |   |       |   |   |   |-- ✅ req_command.py
|   |   |       |   |   |   |-- ✅ spinners.py
|   |   |       |   |   |   \-- ✅ status_codes.py
|   |   |       |   |   |-- ✅ commands/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ cache.py
|   |   |       |   |   |   |-- ✅ check.py
|   |   |       |   |   |   |-- ✅ completion.py
|   |   |       |   |   |   |-- ✅ configuration.py
|   |   |       |   |   |   |-- ✅ debug.py
|   |   |       |   |   |   |-- ✅ download.py
|   |   |       |   |   |   |-- ✅ freeze.py
|   |   |       |   |   |   |-- ✅ hash.py
|   |   |       |   |   |   |-- ✅ help.py
|   |   |       |   |   |   |-- ✅ index.py
|   |   |       |   |   |   |-- ✅ inspect.py
|   |   |       |   |   |   |-- ✅ install.py
|   |   |       |   |   |   |-- ✅ list.py
|   |   |       |   |   |   |-- ✅ lock.py
|   |   |       |   |   |   |-- ✅ search.py
|   |   |       |   |   |   |-- ✅ show.py
|   |   |       |   |   |   |-- ✅ uninstall.py
|   |   |       |   |   |   \-- ✅ wheel.py
|   |   |       |   |   |-- ✅ distributions/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ installed.py
|   |   |       |   |   |   |-- ✅ sdist.py
|   |   |       |   |   |   \-- ✅ wheel.py
|   |   |       |   |   |-- ✅ index/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ collector.py
|   |   |       |   |   |   |-- ✅ package_finder.py
|   |   |       |   |   |   \-- ✅ sources.py
|   |   |       |   |   |-- ✅ locations/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _distutils.py
|   |   |       |   |   |   |-- ✅ _sysconfig.py
|   |   |       |   |   |   \-- ✅ base.py
|   |   |       |   |   |-- ✅ metadata/
|   |   |       |   |   |   |-- ✅ importlib/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _compat.py
|   |   |       |   |   |   |   |-- ✅ _dists.py
|   |   |       |   |   |   |   \-- ✅ _envs.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _json.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   \-- ✅ pkg_resources.py
|   |   |       |   |   |-- ✅ models/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ candidate.py
|   |   |       |   |   |   |-- ✅ direct_url.py
|   |   |       |   |   |   |-- ✅ format_control.py
|   |   |       |   |   |   |-- ✅ index.py
|   |   |       |   |   |   |-- ✅ installation_report.py
|   |   |       |   |   |   |-- ✅ link.py
|   |   |       |   |   |   |-- ✅ pylock.py
|   |   |       |   |   |   |-- ✅ scheme.py
|   |   |       |   |   |   |-- ✅ search_scope.py
|   |   |       |   |   |   |-- ✅ selection_prefs.py
|   |   |       |   |   |   |-- ✅ target_python.py
|   |   |       |   |   |   \-- ✅ wheel.py
|   |   |       |   |   |-- ✅ network/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ auth.py
|   |   |       |   |   |   |-- ✅ cache.py
|   |   |       |   |   |   |-- ✅ download.py
|   |   |       |   |   |   |-- ✅ lazy_wheel.py
|   |   |       |   |   |   |-- ✅ session.py
|   |   |       |   |   |   |-- ✅ utils.py
|   |   |       |   |   |   \-- ✅ xmlrpc.py
|   |   |       |   |   |-- ✅ operations/
|   |   |       |   |   |   |-- ✅ build/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ build_tracker.py
|   |   |       |   |   |   |   |-- ✅ metadata.py
|   |   |       |   |   |   |   |-- ✅ metadata_editable.py
|   |   |       |   |   |   |   |-- ✅ metadata_legacy.py
|   |   |       |   |   |   |   |-- ✅ wheel.py
|   |   |       |   |   |   |   |-- ✅ wheel_editable.py
|   |   |       |   |   |   |   \-- ✅ wheel_legacy.py
|   |   |       |   |   |   |-- ✅ install/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ editable_legacy.py
|   |   |       |   |   |   |   \-- ✅ wheel.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ check.py
|   |   |       |   |   |   |-- ✅ freeze.py
|   |   |       |   |   |   \-- ✅ prepare.py
|   |   |       |   |   |-- ✅ req/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ constructors.py
|   |   |       |   |   |   |-- ✅ req_dependency_group.py
|   |   |       |   |   |   |-- ✅ req_file.py
|   |   |       |   |   |   |-- ✅ req_install.py
|   |   |       |   |   |   |-- ✅ req_set.py
|   |   |       |   |   |   \-- ✅ req_uninstall.py
|   |   |       |   |   |-- ✅ resolution/
|   |   |       |   |   |   |-- ✅ legacy/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ resolver.py
|   |   |       |   |   |   |-- ✅ resolvelib/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |   |-- ✅ candidates.py
|   |   |       |   |   |   |   |-- ✅ factory.py
|   |   |       |   |   |   |   |-- ✅ found_candidates.py
|   |   |       |   |   |   |   |-- ✅ provider.py
|   |   |       |   |   |   |   |-- ✅ reporter.py
|   |   |       |   |   |   |   |-- ✅ requirements.py
|   |   |       |   |   |   |   \-- ✅ resolver.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ base.py
|   |   |       |   |   |-- ✅ utils/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _jaraco_text.py
|   |   |       |   |   |   |-- ✅ _log.py
|   |   |       |   |   |   |-- ✅ appdirs.py
|   |   |       |   |   |   |-- ✅ compat.py
|   |   |       |   |   |   |-- ✅ compatibility_tags.py
|   |   |       |   |   |   |-- ✅ datetime.py
|   |   |       |   |   |   |-- ✅ deprecation.py
|   |   |       |   |   |   |-- ✅ direct_url_helpers.py
|   |   |       |   |   |   |-- ✅ egg_link.py
|   |   |       |   |   |   |-- ✅ entrypoints.py
|   |   |       |   |   |   |-- ✅ filesystem.py
|   |   |       |   |   |   |-- ✅ filetypes.py
|   |   |       |   |   |   |-- ✅ glibc.py
|   |   |       |   |   |   |-- ✅ hashes.py
|   |   |       |   |   |   |-- ✅ logging.py
|   |   |       |   |   |   |-- ✅ misc.py
|   |   |       |   |   |   |-- ✅ packaging.py
|   |   |       |   |   |   |-- ✅ retry.py
|   |   |       |   |   |   |-- ✅ setuptools_build.py
|   |   |       |   |   |   |-- ✅ subprocess.py
|   |   |       |   |   |   |-- ✅ temp_dir.py
|   |   |       |   |   |   |-- ✅ unpacking.py
|   |   |       |   |   |   |-- ✅ urls.py
|   |   |       |   |   |   |-- ✅ virtualenv.py
|   |   |       |   |   |   \-- ✅ wheel.py
|   |   |       |   |   |-- ✅ vcs/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ bazaar.py
|   |   |       |   |   |   |-- ✅ git.py
|   |   |       |   |   |   |-- ✅ mercurial.py
|   |   |       |   |   |   |-- ✅ subversion.py
|   |   |       |   |   |   \-- ✅ versioncontrol.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ build_env.py
|   |   |       |   |   |-- ✅ cache.py
|   |   |       |   |   |-- ✅ configuration.py
|   |   |       |   |   |-- ✅ exceptions.py
|   |   |       |   |   |-- ✅ main.py
|   |   |       |   |   |-- ✅ pyproject.py
|   |   |       |   |   |-- ✅ self_outdated_check.py
|   |   |       |   |   \-- ✅ wheel_builder.py
|   |   |       |   |-- ✅ _vendor/
|   |   |       |   |   |-- ✅ cachecontrol/
|   |   |       |   |   |   |-- ✅ caches/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ file_cache.py
|   |   |       |   |   |   |   \-- ✅ redis_cache.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _cmd.py
|   |   |       |   |   |   |-- ✅ adapter.py
|   |   |       |   |   |   |-- ✅ cache.py
|   |   |       |   |   |   |-- ✅ controller.py
|   |   |       |   |   |   |-- ✅ filewrapper.py
|   |   |       |   |   |   |-- ✅ heuristics.py
|   |   |       |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   |-- ✅ serialize.py
|   |   |       |   |   |   \-- ✅ wrapper.py
|   |   |       |   |   |-- ✅ certifi/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __main__.py
|   |   |       |   |   |   |-- ✅ cacert.pem
|   |   |       |   |   |   |-- ✅ core.py
|   |   |       |   |   |   \-- ✅ py.typed
|   |   |       |   |   |-- ✅ dependency_groups/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __main__.py
|   |   |       |   |   |   |-- ✅ _implementation.py
|   |   |       |   |   |   |-- ✅ _lint_dependency_groups.py
|   |   |       |   |   |   |-- ✅ _pip_wrapper.py
|   |   |       |   |   |   |-- ✅ _toml_compat.py
|   |   |       |   |   |   \-- ✅ py.typed
|   |   |       |   |   |-- ✅ distlib/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ compat.py
|   |   |       |   |   |   |-- ✅ resources.py
|   |   |       |   |   |   |-- ✅ scripts.py
|   |   |       |   |   |   |-- ✅ t32.exe
|   |   |       |   |   |   |-- ✅ t64-arm.exe
|   |   |       |   |   |   |-- ✅ t64.exe
|   |   |       |   |   |   |-- ✅ util.py
|   |   |       |   |   |   |-- ✅ w32.exe
|   |   |       |   |   |   |-- ✅ w64-arm.exe
|   |   |       |   |   |   \-- ✅ w64.exe
|   |   |       |   |   |-- ✅ distro/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __main__.py
|   |   |       |   |   |   |-- ✅ distro.py
|   |   |       |   |   |   \-- ✅ py.typed
|   |   |       |   |   |-- ✅ idna/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ codec.py
|   |   |       |   |   |   |-- ✅ compat.py
|   |   |       |   |   |   |-- ✅ core.py
|   |   |       |   |   |   |-- ✅ idnadata.py
|   |   |       |   |   |   |-- ✅ intranges.py
|   |   |       |   |   |   |-- ✅ package_data.py
|   |   |       |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   \-- ✅ uts46data.py
|   |   |       |   |   |-- ✅ msgpack/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |-- ✅ ext.py
|   |   |       |   |   |   \-- ✅ fallback.py
|   |   |       |   |   |-- ✅ packaging/
|   |   |       |   |   |   |-- ✅ licenses/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ _spdx.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _elffile.py
|   |   |       |   |   |   |-- ✅ _manylinux.py
|   |   |       |   |   |   |-- ✅ _musllinux.py
|   |   |       |   |   |   |-- ✅ _parser.py
|   |   |       |   |   |   |-- ✅ _structures.py
|   |   |       |   |   |   |-- ✅ _tokenizer.py
|   |   |       |   |   |   |-- ✅ markers.py
|   |   |       |   |   |   |-- ✅ metadata.py
|   |   |       |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   |-- ✅ requirements.py
|   |   |       |   |   |   |-- ✅ specifiers.py
|   |   |       |   |   |   |-- ✅ tags.py
|   |   |       |   |   |   |-- ✅ utils.py
|   |   |       |   |   |   \-- ✅ version.py
|   |   |       |   |   |-- ✅ pkg_resources/
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ platformdirs/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __main__.py
|   |   |       |   |   |   |-- ✅ android.py
|   |   |       |   |   |   |-- ✅ api.py
|   |   |       |   |   |   |-- ✅ macos.py
|   |   |       |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   |-- ✅ unix.py
|   |   |       |   |   |   |-- ✅ version.py
|   |   |       |   |   |   \-- ✅ windows.py
|   |   |       |   |   |-- ✅ pygments/
|   |   |       |   |   |   |-- ✅ filters/
|   |   |       |   |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ formatters/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ _mapping.py
|   |   |       |   |   |   |-- ✅ lexers/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _mapping.py
|   |   |       |   |   |   |   \-- ✅ python.py
|   |   |       |   |   |   |-- ✅ styles/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ _mapping.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __main__.py
|   |   |       |   |   |   |-- ✅ console.py
|   |   |       |   |   |   |-- ✅ filter.py
|   |   |       |   |   |   |-- ✅ formatter.py
|   |   |       |   |   |   |-- ✅ lexer.py
|   |   |       |   |   |   |-- ✅ modeline.py
|   |   |       |   |   |   |-- ✅ plugin.py
|   |   |       |   |   |   |-- ✅ regexopt.py
|   |   |       |   |   |   |-- ✅ scanner.py
|   |   |       |   |   |   |-- ✅ sphinxext.py
|   |   |       |   |   |   |-- ✅ style.py
|   |   |       |   |   |   |-- ✅ token.py
|   |   |       |   |   |   |-- ✅ unistring.py
|   |   |       |   |   |   \-- ✅ util.py
|   |   |       |   |   |-- ✅ pyproject_hooks/
|   |   |       |   |   |   |-- ✅ _in_process/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ _in_process.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _impl.py
|   |   |       |   |   |   \-- ✅ py.typed
|   |   |       |   |   |-- ✅ requests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __version__.py
|   |   |       |   |   |   |-- ✅ _internal_utils.py
|   |   |       |   |   |   |-- ✅ adapters.py
|   |   |       |   |   |   |-- ✅ api.py
|   |   |       |   |   |   |-- ✅ auth.py
|   |   |       |   |   |   |-- ✅ certs.py
|   |   |       |   |   |   |-- ✅ compat.py
|   |   |       |   |   |   |-- ✅ cookies.py
|   |   |       |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |-- ✅ help.py
|   |   |       |   |   |   |-- ✅ hooks.py
|   |   |       |   |   |   |-- ✅ models.py
|   |   |       |   |   |   |-- ✅ packages.py
|   |   |       |   |   |   |-- ✅ sessions.py
|   |   |       |   |   |   |-- ✅ status_codes.py
|   |   |       |   |   |   |-- ✅ structures.py
|   |   |       |   |   |   \-- ✅ utils.py
|   |   |       |   |   |-- ✅ resolvelib/
|   |   |       |   |   |   |-- ✅ resolvers/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ abstract.py
|   |   |       |   |   |   |   |-- ✅ criterion.py
|   |   |       |   |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |   \-- ✅ resolution.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ providers.py
|   |   |       |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   |-- ✅ reporters.py
|   |   |       |   |   |   \-- ✅ structs.py
|   |   |       |   |   |-- ✅ rich/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ __main__.py
|   |   |       |   |   |   |-- ✅ _cell_widths.py
|   |   |       |   |   |   |-- ✅ _emoji_codes.py
|   |   |       |   |   |   |-- ✅ _emoji_replace.py
|   |   |       |   |   |   |-- ✅ _export_format.py
|   |   |       |   |   |   |-- ✅ _extension.py
|   |   |       |   |   |   |-- ✅ _fileno.py
|   |   |       |   |   |   |-- ✅ _inspect.py
|   |   |       |   |   |   |-- ✅ _log_render.py
|   |   |       |   |   |   |-- ✅ _loop.py
|   |   |       |   |   |   |-- ✅ _null_file.py
|   |   |       |   |   |   |-- ✅ _palettes.py
|   |   |       |   |   |   |-- ✅ _pick.py
|   |   |       |   |   |   |-- ✅ _ratio.py
|   |   |       |   |   |   |-- ✅ _spinners.py
|   |   |       |   |   |   |-- ✅ _stack.py
|   |   |       |   |   |   |-- ✅ _timer.py
|   |   |       |   |   |   |-- ✅ _win32_console.py
|   |   |       |   |   |   |-- ✅ _windows.py
|   |   |       |   |   |   |-- ✅ _windows_renderer.py
|   |   |       |   |   |   |-- ✅ _wrap.py
|   |   |       |   |   |   |-- ✅ abc.py
|   |   |       |   |   |   |-- ✅ align.py
|   |   |       |   |   |   |-- ✅ ansi.py
|   |   |       |   |   |   |-- ✅ bar.py
|   |   |       |   |   |   |-- ✅ box.py
|   |   |       |   |   |   |-- ✅ cells.py
|   |   |       |   |   |   |-- ✅ color.py
|   |   |       |   |   |   |-- ✅ color_triplet.py
|   |   |       |   |   |   |-- ✅ columns.py
|   |   |       |   |   |   |-- ✅ console.py
|   |   |       |   |   |   |-- ✅ constrain.py
|   |   |       |   |   |   |-- ✅ containers.py
|   |   |       |   |   |   |-- ✅ control.py
|   |   |       |   |   |   |-- ✅ default_styles.py
|   |   |       |   |   |   |-- ✅ diagnose.py
|   |   |       |   |   |   |-- ✅ emoji.py
|   |   |       |   |   |   |-- ✅ errors.py
|   |   |       |   |   |   |-- ✅ file_proxy.py
|   |   |       |   |   |   |-- ✅ filesize.py
|   |   |       |   |   |   |-- ✅ highlighter.py
|   |   |       |   |   |   |-- ✅ json.py
|   |   |       |   |   |   |-- ✅ jupyter.py
|   |   |       |   |   |   |-- ✅ layout.py
|   |   |       |   |   |   |-- ✅ live.py
|   |   |       |   |   |   |-- ✅ live_render.py
|   |   |       |   |   |   |-- ✅ logging.py
|   |   |       |   |   |   |-- ✅ markup.py
|   |   |       |   |   |   |-- ✅ measure.py
|   |   |       |   |   |   |-- ✅ padding.py
|   |   |       |   |   |   |-- ✅ pager.py
|   |   |       |   |   |   |-- ✅ palette.py
|   |   |       |   |   |   |-- ✅ panel.py
|   |   |       |   |   |   |-- ✅ pretty.py
|   |   |       |   |   |   |-- ✅ progress.py
|   |   |       |   |   |   |-- ✅ progress_bar.py
|   |   |       |   |   |   |-- ✅ prompt.py
|   |   |       |   |   |   |-- ✅ protocol.py
|   |   |       |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   |-- ✅ region.py
|   |   |       |   |   |   |-- ✅ repr.py
|   |   |       |   |   |   |-- ✅ rule.py
|   |   |       |   |   |   |-- ✅ scope.py
|   |   |       |   |   |   |-- ✅ screen.py
|   |   |       |   |   |   |-- ✅ segment.py
|   |   |       |   |   |   |-- ✅ spinner.py
|   |   |       |   |   |   |-- ✅ status.py
|   |   |       |   |   |   |-- ✅ style.py
|   |   |       |   |   |   |-- ✅ styled.py
|   |   |       |   |   |   |-- ✅ syntax.py
|   |   |       |   |   |   |-- ✅ table.py
|   |   |       |   |   |   |-- ✅ terminal_theme.py
|   |   |       |   |   |   |-- ✅ text.py
|   |   |       |   |   |   |-- ✅ theme.py
|   |   |       |   |   |   |-- ✅ themes.py
|   |   |       |   |   |   |-- ✅ traceback.py
|   |   |       |   |   |   \-- ✅ tree.py
|   |   |       |   |   |-- ✅ tomli/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _parser.py
|   |   |       |   |   |   |-- ✅ _re.py
|   |   |       |   |   |   |-- ✅ _types.py
|   |   |       |   |   |   \-- ✅ py.typed
|   |   |       |   |   |-- ✅ tomli_w/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _writer.py
|   |   |       |   |   |   \-- ✅ py.typed
|   |   |       |   |   |-- ✅ truststore/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _api.py
|   |   |       |   |   |   |-- ✅ _macos.py
|   |   |       |   |   |   |-- ✅ _openssl.py
|   |   |       |   |   |   |-- ✅ _ssl_constants.py
|   |   |       |   |   |   |-- ✅ _windows.py
|   |   |       |   |   |   \-- ✅ py.typed
|   |   |       |   |   |-- ✅ urllib3/
|   |   |       |   |   |   |-- ✅ contrib/
|   |   |       |   |   |   |   |-- ✅ _securetransport/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ bindings.py
|   |   |       |   |   |   |   |   \-- ✅ low_level.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _appengine_environ.py
|   |   |       |   |   |   |   |-- ✅ appengine.py
|   |   |       |   |   |   |   |-- ✅ ntlmpool.py
|   |   |       |   |   |   |   |-- ✅ pyopenssl.py
|   |   |       |   |   |   |   |-- ✅ securetransport.py
|   |   |       |   |   |   |   \-- ✅ socks.py
|   |   |       |   |   |   |-- ✅ packages/
|   |   |       |   |   |   |   |-- ✅ backports/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ makefile.py
|   |   |       |   |   |   |   |   \-- ✅ weakref_finalize.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ six.py
|   |   |       |   |   |   |-- ✅ util/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ connection.py
|   |   |       |   |   |   |   |-- ✅ proxy.py
|   |   |       |   |   |   |   |-- ✅ queue.py
|   |   |       |   |   |   |   |-- ✅ request.py
|   |   |       |   |   |   |   |-- ✅ response.py
|   |   |       |   |   |   |   |-- ✅ retry.py
|   |   |       |   |   |   |   |-- ✅ ssl_.py
|   |   |       |   |   |   |   |-- ✅ ssl_match_hostname.py
|   |   |       |   |   |   |   |-- ✅ ssltransport.py
|   |   |       |   |   |   |   |-- ✅ timeout.py
|   |   |       |   |   |   |   |-- ✅ url.py
|   |   |       |   |   |   |   \-- ✅ wait.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _collections.py
|   |   |       |   |   |   |-- ✅ _version.py
|   |   |       |   |   |   |-- ✅ connection.py
|   |   |       |   |   |   |-- ✅ connectionpool.py
|   |   |       |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |-- ✅ fields.py
|   |   |       |   |   |   |-- ✅ filepost.py
|   |   |       |   |   |   |-- ✅ poolmanager.py
|   |   |       |   |   |   |-- ✅ request.py
|   |   |       |   |   |   \-- ✅ response.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ vendor.txt
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __main__.py
|   |   |       |   |-- ✅ __pip-runner__.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ pip-25.2.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ src/
|   |   |       |   |   |   \-- ✅ pip/
|   |   |       |   |   |       \-- ✅ _vendor/
|   |   |       |   |   |           |-- ✅ cachecontrol/
|   |   |       |   |   |           |   \-- ✅ LICENSE.txt
|   |   |       |   |   |           |-- ✅ certifi/
|   |   |       |   |   |           |   \-- ✅ LICENSE
|   |   |       |   |   |           |-- ✅ dependency_groups/
|   |   |       |   |   |           |   \-- ✅ LICENSE.txt
|   |   |       |   |   |           |-- ✅ distlib/
|   |   |       |   |   |           |   \-- ✅ LICENSE.txt
|   |   |       |   |   |           |-- ✅ distro/
|   |   |       |   |   |           |   \-- ✅ LICENSE
|   |   |       |   |   |           |-- ✅ idna/
|   |   |       |   |   |           |   \-- ✅ LICENSE.md
|   |   |       |   |   |           |-- ✅ msgpack/
|   |   |       |   |   |           |   \-- ✅ COPYING
|   |   |       |   |   |           |-- ✅ packaging/
|   |   |       |   |   |           |   |-- ✅ LICENSE
|   |   |       |   |   |           |   |-- ✅ LICENSE.APACHE
|   |   |       |   |   |           |   \-- ✅ LICENSE.BSD
|   |   |       |   |   |           |-- ✅ pkg_resources/
|   |   |       |   |   |           |   \-- ✅ LICENSE
|   |   |       |   |   |           |-- ✅ platformdirs/
|   |   |       |   |   |           |   \-- ✅ LICENSE
|   |   |       |   |   |           |-- ✅ pygments/
|   |   |       |   |   |           |   \-- ✅ LICENSE
|   |   |       |   |   |           |-- ✅ pyproject_hooks/
|   |   |       |   |   |           |   \-- ✅ LICENSE
|   |   |       |   |   |           |-- ✅ requests/
|   |   |       |   |   |           |   \-- ✅ LICENSE
|   |   |       |   |   |           |-- ✅ resolvelib/
|   |   |       |   |   |           |   \-- ✅ LICENSE
|   |   |       |   |   |           |-- ✅ rich/
|   |   |       |   |   |           |   \-- ✅ LICENSE
|   |   |       |   |   |           |-- ✅ tomli/
|   |   |       |   |   |           |   |-- ✅ LICENSE
|   |   |       |   |   |           |   \-- ✅ LICENSE-HEADER
|   |   |       |   |   |           |-- ✅ tomli_w/
|   |   |       |   |   |           |   \-- ✅ LICENSE
|   |   |       |   |   |           |-- ✅ truststore/
|   |   |       |   |   |           |   \-- ✅ LICENSE
|   |   |       |   |   |           \-- ✅ urllib3/
|   |   |       |   |   |               \-- ✅ LICENSE.txt
|   |   |       |   |   |-- ✅ AUTHORS.txt
|   |   |       |   |   \-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ propcache/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _helpers.py
|   |   |       |   |-- ✅ _helpers_c.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _helpers_c.pyx
|   |   |       |   |-- ✅ _helpers_py.py
|   |   |       |   |-- ✅ api.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ propcache-0.3.2.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ LICENSE
|   |   |       |   |   \-- ✅ NOTICE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ pycparser/
|   |   |       |   |-- ✅ ply/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ cpp.py
|   |   |       |   |   |-- ✅ ctokens.py
|   |   |       |   |   |-- ✅ lex.py
|   |   |       |   |   |-- ✅ yacc.py
|   |   |       |   |   \-- ✅ ygen.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _ast_gen.py
|   |   |       |   |-- ✅ _build_tables.py
|   |   |       |   |-- ✅ _c_ast.cfg
|   |   |       |   |-- ✅ ast_transforms.py
|   |   |       |   |-- ✅ c_ast.py
|   |   |       |   |-- ✅ c_generator.py
|   |   |       |   |-- ✅ c_lexer.py
|   |   |       |   |-- ✅ c_parser.py
|   |   |       |   |-- ✅ lextab.py
|   |   |       |   |-- ✅ plyparser.py
|   |   |       |   \-- ✅ yacctab.py
|   |   |       |-- ✅ pycparser-2.23.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ pycryptodome-3.23.0.dist-info/
|   |   |       |   |-- ✅ AUTHORS.rst
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE.rst
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ pydantic/
|   |   |       |   |-- ✅ _internal/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _config.py
|   |   |       |   |   |-- ✅ _core_metadata.py
|   |   |       |   |   |-- ✅ _core_utils.py
|   |   |       |   |   |-- ✅ _dataclasses.py
|   |   |       |   |   |-- ✅ _decorators.py
|   |   |       |   |   |-- ✅ _decorators_v1.py
|   |   |       |   |   |-- ✅ _discriminated_union.py
|   |   |       |   |   |-- ✅ _docs_extraction.py
|   |   |       |   |   |-- ✅ _fields.py
|   |   |       |   |   |-- ✅ _forward_ref.py
|   |   |       |   |   |-- ✅ _generate_schema.py
|   |   |       |   |   |-- ✅ _generics.py
|   |   |       |   |   |-- ✅ _git.py
|   |   |       |   |   |-- ✅ _import_utils.py
|   |   |       |   |   |-- ✅ _internal_dataclass.py
|   |   |       |   |   |-- ✅ _known_annotated_metadata.py
|   |   |       |   |   |-- ✅ _mock_val_ser.py
|   |   |       |   |   |-- ✅ _model_construction.py
|   |   |       |   |   |-- ✅ _namespace_utils.py
|   |   |       |   |   |-- ✅ _repr.py
|   |   |       |   |   |-- ✅ _schema_gather.py
|   |   |       |   |   |-- ✅ _schema_generation_shared.py
|   |   |       |   |   |-- ✅ _serializers.py
|   |   |       |   |   |-- ✅ _signature.py
|   |   |       |   |   |-- ✅ _typing_extra.py
|   |   |       |   |   |-- ✅ _utils.py
|   |   |       |   |   |-- ✅ _validate_call.py
|   |   |       |   |   \-- ✅ _validators.py
|   |   |       |   |-- ✅ deprecated/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ class_validators.py
|   |   |       |   |   |-- ✅ config.py
|   |   |       |   |   |-- ✅ copy_internals.py
|   |   |       |   |   |-- ✅ decorator.py
|   |   |       |   |   |-- ✅ json.py
|   |   |       |   |   |-- ✅ parse.py
|   |   |       |   |   \-- ✅ tools.py
|   |   |       |   |-- ✅ experimental/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ arguments_schema.py
|   |   |       |   |   \-- ✅ pipeline.py
|   |   |       |   |-- ✅ plugin/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _loader.py
|   |   |       |   |   \-- ✅ _schema_validator.py
|   |   |       |   |-- ✅ v1/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _hypothesis_plugin.py
|   |   |       |   |   |-- ✅ annotated_types.py
|   |   |       |   |   |-- ✅ class_validators.py
|   |   |       |   |   |-- ✅ color.py
|   |   |       |   |   |-- ✅ config.py
|   |   |       |   |   |-- ✅ dataclasses.py
|   |   |       |   |   |-- ✅ datetime_parse.py
|   |   |       |   |   |-- ✅ decorator.py
|   |   |       |   |   |-- ✅ env_settings.py
|   |   |       |   |   |-- ✅ error_wrappers.py
|   |   |       |   |   |-- ✅ errors.py
|   |   |       |   |   |-- ✅ fields.py
|   |   |       |   |   |-- ✅ generics.py
|   |   |       |   |   |-- ✅ json.py
|   |   |       |   |   |-- ✅ main.py
|   |   |       |   |   |-- ✅ mypy.py
|   |   |       |   |   |-- ✅ networks.py
|   |   |       |   |   |-- ✅ parse.py
|   |   |       |   |   |-- ✅ py.typed
|   |   |       |   |   |-- ✅ schema.py
|   |   |       |   |   |-- ✅ tools.py
|   |   |       |   |   |-- ✅ types.py
|   |   |       |   |   |-- ✅ typing.py
|   |   |       |   |   |-- ✅ utils.py
|   |   |       |   |   |-- ✅ validators.py
|   |   |       |   |   \-- ✅ version.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _migration.py
|   |   |       |   |-- ✅ alias_generators.py
|   |   |       |   |-- ✅ aliases.py
|   |   |       |   |-- ✅ annotated_handlers.py
|   |   |       |   |-- ✅ class_validators.py
|   |   |       |   |-- ✅ color.py
|   |   |       |   |-- ✅ config.py
|   |   |       |   |-- ✅ dataclasses.py
|   |   |       |   |-- ✅ datetime_parse.py
|   |   |       |   |-- ✅ decorator.py
|   |   |       |   |-- ✅ env_settings.py
|   |   |       |   |-- ✅ error_wrappers.py
|   |   |       |   |-- ✅ errors.py
|   |   |       |   |-- ✅ fields.py
|   |   |       |   |-- ✅ functional_serializers.py
|   |   |       |   |-- ✅ functional_validators.py
|   |   |       |   |-- ✅ generics.py
|   |   |       |   |-- ✅ json.py
|   |   |       |   |-- ✅ json_schema.py
|   |   |       |   |-- ✅ main.py
|   |   |       |   |-- ✅ mypy.py
|   |   |       |   |-- ✅ networks.py
|   |   |       |   |-- ✅ parse.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ root_model.py
|   |   |       |   |-- ✅ schema.py
|   |   |       |   |-- ✅ tools.py
|   |   |       |   |-- ✅ type_adapter.py
|   |   |       |   |-- ✅ types.py
|   |   |       |   |-- ✅ typing.py
|   |   |       |   |-- ✅ utils.py
|   |   |       |   |-- ✅ validate_call_decorator.py
|   |   |       |   |-- ✅ validators.py
|   |   |       |   |-- ✅ version.py
|   |   |       |   \-- ✅ warnings.py
|   |   |       |-- ✅ pydantic-2.11.9.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ pydantic_core/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _pydantic_core.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _pydantic_core.pyi
|   |   |       |   |-- ✅ core_schema.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ pydantic_core-2.33.2.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ pyparsing/
|   |   |       |   |-- ✅ diagram/
|   |   |       |   |   \-- ✅ __init__.py
|   |   |       |   |-- ✅ tools/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ cvt_pyparsing_pep8_names.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ actions.py
|   |   |       |   |-- ✅ common.py
|   |   |       |   |-- ✅ core.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ helpers.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ results.py
|   |   |       |   |-- ✅ testing.py
|   |   |       |   |-- ✅ unicode.py
|   |   |       |   \-- ✅ util.py
|   |   |       |-- ✅ pyparsing-3.2.5.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ python_binance-1.0.29.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ python_dateutil-2.9.0.post0.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   |-- ✅ WHEEL
|   |   |       |   \-- ✅ zip-safe
|   |   |       |-- ✅ python_dotenv-1.1.1.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ pytz/
|   |   |       |   |-- ✅ zoneinfo/
|   |   |       |   |   |-- ✅ Africa/
|   |   |       |   |   |   |-- ✅ Abidjan
|   |   |       |   |   |   |-- ✅ Accra
|   |   |       |   |   |   |-- ✅ Addis_Ababa
|   |   |       |   |   |   |-- ✅ Algiers
|   |   |       |   |   |   |-- ✅ Asmara
|   |   |       |   |   |   |-- ✅ Asmera
|   |   |       |   |   |   |-- ✅ Bamako
|   |   |       |   |   |   |-- ✅ Bangui
|   |   |       |   |   |   |-- ✅ Banjul
|   |   |       |   |   |   |-- ✅ Bissau
|   |   |       |   |   |   |-- ✅ Blantyre
|   |   |       |   |   |   |-- ✅ Brazzaville
|   |   |       |   |   |   |-- ✅ Bujumbura
|   |   |       |   |   |   |-- ✅ Cairo
|   |   |       |   |   |   |-- ✅ Casablanca
|   |   |       |   |   |   |-- ✅ Ceuta
|   |   |       |   |   |   |-- ✅ Conakry
|   |   |       |   |   |   |-- ✅ Dakar
|   |   |       |   |   |   |-- ✅ Dar_es_Salaam
|   |   |       |   |   |   |-- ✅ Djibouti
|   |   |       |   |   |   |-- ✅ Douala
|   |   |       |   |   |   |-- ✅ El_Aaiun
|   |   |       |   |   |   |-- ✅ Freetown
|   |   |       |   |   |   |-- ✅ Gaborone
|   |   |       |   |   |   |-- ✅ Harare
|   |   |       |   |   |   |-- ✅ Johannesburg
|   |   |       |   |   |   |-- ✅ Juba
|   |   |       |   |   |   |-- ✅ Kampala
|   |   |       |   |   |   |-- ✅ Khartoum
|   |   |       |   |   |   |-- ✅ Kigali
|   |   |       |   |   |   |-- ✅ Kinshasa
|   |   |       |   |   |   |-- ✅ Lagos
|   |   |       |   |   |   |-- ✅ Libreville
|   |   |       |   |   |   |-- ✅ Lome
|   |   |       |   |   |   |-- ✅ Luanda
|   |   |       |   |   |   |-- ✅ Lubumbashi
|   |   |       |   |   |   |-- ✅ Lusaka
|   |   |       |   |   |   |-- ✅ Malabo
|   |   |       |   |   |   |-- ✅ Maputo
|   |   |       |   |   |   |-- ✅ Maseru
|   |   |       |   |   |   |-- ✅ Mbabane
|   |   |       |   |   |   |-- ✅ Mogadishu
|   |   |       |   |   |   |-- ✅ Monrovia
|   |   |       |   |   |   |-- ✅ Nairobi
|   |   |       |   |   |   |-- ✅ Ndjamena
|   |   |       |   |   |   |-- ✅ Niamey
|   |   |       |   |   |   |-- ✅ Nouakchott
|   |   |       |   |   |   |-- ✅ Ouagadougou
|   |   |       |   |   |   |-- ✅ Porto-Novo
|   |   |       |   |   |   |-- ✅ Sao_Tome
|   |   |       |   |   |   |-- ✅ Timbuktu
|   |   |       |   |   |   |-- ✅ Tripoli
|   |   |       |   |   |   |-- ✅ Tunis
|   |   |       |   |   |   \-- ✅ Windhoek
|   |   |       |   |   |-- ✅ America/
|   |   |       |   |   |   |-- ✅ Argentina/
|   |   |       |   |   |   |   |-- ✅ Buenos_Aires
|   |   |       |   |   |   |   |-- ✅ Catamarca
|   |   |       |   |   |   |   |-- ✅ ComodRivadavia
|   |   |       |   |   |   |   |-- ✅ Cordoba
|   |   |       |   |   |   |   |-- ✅ Jujuy
|   |   |       |   |   |   |   |-- ✅ La_Rioja
|   |   |       |   |   |   |   |-- ✅ Mendoza
|   |   |       |   |   |   |   |-- ✅ Rio_Gallegos
|   |   |       |   |   |   |   |-- ✅ Salta
|   |   |       |   |   |   |   |-- ✅ San_Juan
|   |   |       |   |   |   |   |-- ✅ San_Luis
|   |   |       |   |   |   |   |-- ✅ Tucuman
|   |   |       |   |   |   |   \-- ✅ Ushuaia
|   |   |       |   |   |   |-- ✅ Indiana/
|   |   |       |   |   |   |   |-- ✅ Indianapolis
|   |   |       |   |   |   |   |-- ✅ Knox
|   |   |       |   |   |   |   |-- ✅ Marengo
|   |   |       |   |   |   |   |-- ✅ Petersburg
|   |   |       |   |   |   |   |-- ✅ Tell_City
|   |   |       |   |   |   |   |-- ✅ Vevay
|   |   |       |   |   |   |   |-- ✅ Vincennes
|   |   |       |   |   |   |   \-- ✅ Winamac
|   |   |       |   |   |   |-- ✅ Kentucky/
|   |   |       |   |   |   |   |-- ✅ Louisville
|   |   |       |   |   |   |   \-- ✅ Monticello
|   |   |       |   |   |   |-- ✅ North_Dakota/
|   |   |       |   |   |   |   |-- ✅ Beulah
|   |   |       |   |   |   |   |-- ✅ Center
|   |   |       |   |   |   |   \-- ✅ New_Salem
|   |   |       |   |   |   |-- ✅ Adak
|   |   |       |   |   |   |-- ✅ Anchorage
|   |   |       |   |   |   |-- ✅ Anguilla
|   |   |       |   |   |   |-- ✅ Antigua
|   |   |       |   |   |   |-- ✅ Araguaina
|   |   |       |   |   |   |-- ✅ Aruba
|   |   |       |   |   |   |-- ✅ Asuncion
|   |   |       |   |   |   |-- ✅ Atikokan
|   |   |       |   |   |   |-- ✅ Atka
|   |   |       |   |   |   |-- ✅ Bahia
|   |   |       |   |   |   |-- ✅ Bahia_Banderas
|   |   |       |   |   |   |-- ✅ Barbados
|   |   |       |   |   |   |-- ✅ Belem
|   |   |       |   |   |   |-- ✅ Belize
|   |   |       |   |   |   |-- ✅ Blanc-Sablon
|   |   |       |   |   |   |-- ✅ Boa_Vista
|   |   |       |   |   |   |-- ✅ Bogota
|   |   |       |   |   |   |-- ✅ Boise
|   |   |       |   |   |   |-- ✅ Buenos_Aires
|   |   |       |   |   |   |-- ✅ Cambridge_Bay
|   |   |       |   |   |   |-- ✅ Campo_Grande
|   |   |       |   |   |   |-- ✅ Cancun
|   |   |       |   |   |   |-- ✅ Caracas
|   |   |       |   |   |   |-- ✅ Catamarca
|   |   |       |   |   |   |-- ✅ Cayenne
|   |   |       |   |   |   |-- ✅ Cayman
|   |   |       |   |   |   |-- ✅ Chicago
|   |   |       |   |   |   |-- ✅ Chihuahua
|   |   |       |   |   |   |-- ✅ Ciudad_Juarez
|   |   |       |   |   |   |-- ✅ Coral_Harbour
|   |   |       |   |   |   |-- ✅ Cordoba
|   |   |       |   |   |   |-- ✅ Costa_Rica
|   |   |       |   |   |   |-- ✅ Coyhaique
|   |   |       |   |   |   |-- ✅ Creston
|   |   |       |   |   |   |-- ✅ Cuiaba
|   |   |       |   |   |   |-- ✅ Curacao
|   |   |       |   |   |   |-- ✅ Danmarkshavn
|   |   |       |   |   |   |-- ✅ Dawson
|   |   |       |   |   |   |-- ✅ Dawson_Creek
|   |   |       |   |   |   |-- ✅ Denver
|   |   |       |   |   |   |-- ✅ Detroit
|   |   |       |   |   |   |-- ✅ Dominica
|   |   |       |   |   |   |-- ✅ Edmonton
|   |   |       |   |   |   |-- ✅ Eirunepe
|   |   |       |   |   |   |-- ✅ El_Salvador
|   |   |       |   |   |   |-- ✅ Ensenada
|   |   |       |   |   |   |-- ✅ Fort_Nelson
|   |   |       |   |   |   |-- ✅ Fort_Wayne
|   |   |       |   |   |   |-- ✅ Fortaleza
|   |   |       |   |   |   |-- ✅ Glace_Bay
|   |   |       |   |   |   |-- ✅ Godthab
|   |   |       |   |   |   |-- ✅ Goose_Bay
|   |   |       |   |   |   |-- ✅ Grand_Turk
|   |   |       |   |   |   |-- ✅ Grenada
|   |   |       |   |   |   |-- ✅ Guadeloupe
|   |   |       |   |   |   |-- ✅ Guatemala
|   |   |       |   |   |   |-- ✅ Guayaquil
|   |   |       |   |   |   |-- ✅ Guyana
|   |   |       |   |   |   |-- ✅ Halifax
|   |   |       |   |   |   |-- ✅ Havana
|   |   |       |   |   |   |-- ✅ Hermosillo
|   |   |       |   |   |   |-- ✅ Indianapolis
|   |   |       |   |   |   |-- ✅ Inuvik
|   |   |       |   |   |   |-- ✅ Iqaluit
|   |   |       |   |   |   |-- ✅ Jamaica
|   |   |       |   |   |   |-- ✅ Jujuy
|   |   |       |   |   |   |-- ✅ Juneau
|   |   |       |   |   |   |-- ✅ Knox_IN
|   |   |       |   |   |   |-- ✅ Kralendijk
|   |   |       |   |   |   |-- ✅ La_Paz
|   |   |       |   |   |   |-- ✅ Lima
|   |   |       |   |   |   |-- ✅ Los_Angeles
|   |   |       |   |   |   |-- ✅ Louisville
|   |   |       |   |   |   |-- ✅ Lower_Princes
|   |   |       |   |   |   |-- ✅ Maceio
|   |   |       |   |   |   |-- ✅ Managua
|   |   |       |   |   |   |-- ✅ Manaus
|   |   |       |   |   |   |-- ✅ Marigot
|   |   |       |   |   |   |-- ✅ Martinique
|   |   |       |   |   |   |-- ✅ Matamoros
|   |   |       |   |   |   |-- ✅ Mazatlan
|   |   |       |   |   |   |-- ✅ Mendoza
|   |   |       |   |   |   |-- ✅ Menominee
|   |   |       |   |   |   |-- ✅ Merida
|   |   |       |   |   |   |-- ✅ Metlakatla
|   |   |       |   |   |   |-- ✅ Mexico_City
|   |   |       |   |   |   |-- ✅ Miquelon
|   |   |       |   |   |   |-- ✅ Moncton
|   |   |       |   |   |   |-- ✅ Monterrey
|   |   |       |   |   |   |-- ✅ Montevideo
|   |   |       |   |   |   |-- ✅ Montreal
|   |   |       |   |   |   |-- ✅ Montserrat
|   |   |       |   |   |   |-- ✅ Nassau
|   |   |       |   |   |   |-- ✅ New_York
|   |   |       |   |   |   |-- ✅ Nipigon
|   |   |       |   |   |   |-- ✅ Nome
|   |   |       |   |   |   |-- ✅ Noronha
|   |   |       |   |   |   |-- ✅ Nuuk
|   |   |       |   |   |   |-- ✅ Ojinaga
|   |   |       |   |   |   |-- ✅ Panama
|   |   |       |   |   |   |-- ✅ Pangnirtung
|   |   |       |   |   |   |-- ✅ Paramaribo
|   |   |       |   |   |   |-- ✅ Phoenix
|   |   |       |   |   |   |-- ✅ Port-au-Prince
|   |   |       |   |   |   |-- ✅ Port_of_Spain
|   |   |       |   |   |   |-- ✅ Porto_Acre
|   |   |       |   |   |   |-- ✅ Porto_Velho
|   |   |       |   |   |   |-- ✅ Puerto_Rico
|   |   |       |   |   |   |-- ✅ Punta_Arenas
|   |   |       |   |   |   |-- ✅ Rainy_River
|   |   |       |   |   |   |-- ✅ Rankin_Inlet
|   |   |       |   |   |   |-- ✅ Recife
|   |   |       |   |   |   |-- ✅ Regina
|   |   |       |   |   |   |-- ✅ Resolute
|   |   |       |   |   |   |-- ✅ Rio_Branco
|   |   |       |   |   |   |-- ✅ Rosario
|   |   |       |   |   |   |-- ✅ Santa_Isabel
|   |   |       |   |   |   |-- ✅ Santarem
|   |   |       |   |   |   |-- ✅ Santiago
|   |   |       |   |   |   |-- ✅ Santo_Domingo
|   |   |       |   |   |   |-- ✅ Sao_Paulo
|   |   |       |   |   |   |-- ✅ Scoresbysund
|   |   |       |   |   |   |-- ✅ Shiprock
|   |   |       |   |   |   |-- ✅ Sitka
|   |   |       |   |   |   |-- ✅ St_Barthelemy
|   |   |       |   |   |   |-- ✅ St_Johns
|   |   |       |   |   |   |-- ✅ St_Kitts
|   |   |       |   |   |   |-- ✅ St_Lucia
|   |   |       |   |   |   |-- ✅ St_Thomas
|   |   |       |   |   |   |-- ✅ St_Vincent
|   |   |       |   |   |   |-- ✅ Swift_Current
|   |   |       |   |   |   |-- ✅ Tegucigalpa
|   |   |       |   |   |   |-- ✅ Thule
|   |   |       |   |   |   |-- ✅ Thunder_Bay
|   |   |       |   |   |   |-- ✅ Tijuana
|   |   |       |   |   |   |-- ✅ Toronto
|   |   |       |   |   |   |-- ✅ Tortola
|   |   |       |   |   |   |-- ✅ Vancouver
|   |   |       |   |   |   |-- ✅ Virgin
|   |   |       |   |   |   |-- ✅ Whitehorse
|   |   |       |   |   |   |-- ✅ Winnipeg
|   |   |       |   |   |   |-- ✅ Yakutat
|   |   |       |   |   |   \-- ✅ Yellowknife
|   |   |       |   |   |-- ✅ Antarctica/
|   |   |       |   |   |   |-- ✅ Casey
|   |   |       |   |   |   |-- ✅ Davis
|   |   |       |   |   |   |-- ✅ DumontDUrville
|   |   |       |   |   |   |-- ✅ Macquarie
|   |   |       |   |   |   |-- ✅ Mawson
|   |   |       |   |   |   |-- ✅ McMurdo
|   |   |       |   |   |   |-- ✅ Palmer
|   |   |       |   |   |   |-- ✅ Rothera
|   |   |       |   |   |   |-- ✅ South_Pole
|   |   |       |   |   |   |-- ✅ Syowa
|   |   |       |   |   |   |-- ✅ Troll
|   |   |       |   |   |   \-- ✅ Vostok
|   |   |       |   |   |-- ✅ Arctic/
|   |   |       |   |   |   \-- ✅ Longyearbyen
|   |   |       |   |   |-- ✅ Asia/
|   |   |       |   |   |   |-- ✅ Aden
|   |   |       |   |   |   |-- ✅ Almaty
|   |   |       |   |   |   |-- ✅ Amman
|   |   |       |   |   |   |-- ✅ Anadyr
|   |   |       |   |   |   |-- ✅ Aqtau
|   |   |       |   |   |   |-- ✅ Aqtobe
|   |   |       |   |   |   |-- ✅ Ashgabat
|   |   |       |   |   |   |-- ✅ Ashkhabad
|   |   |       |   |   |   |-- ✅ Atyrau
|   |   |       |   |   |   |-- ✅ Baghdad
|   |   |       |   |   |   |-- ✅ Bahrain
|   |   |       |   |   |   |-- ✅ Baku
|   |   |       |   |   |   |-- ✅ Bangkok
|   |   |       |   |   |   |-- ✅ Barnaul
|   |   |       |   |   |   |-- ✅ Beirut
|   |   |       |   |   |   |-- ✅ Bishkek
|   |   |       |   |   |   |-- ✅ Brunei
|   |   |       |   |   |   |-- ✅ Calcutta
|   |   |       |   |   |   |-- ✅ Chita
|   |   |       |   |   |   |-- ✅ Choibalsan
|   |   |       |   |   |   |-- ✅ Chongqing
|   |   |       |   |   |   |-- ✅ Chungking
|   |   |       |   |   |   |-- ✅ Colombo
|   |   |       |   |   |   |-- ✅ Dacca
|   |   |       |   |   |   |-- ✅ Damascus
|   |   |       |   |   |   |-- ✅ Dhaka
|   |   |       |   |   |   |-- ✅ Dili
|   |   |       |   |   |   |-- ✅ Dubai
|   |   |       |   |   |   |-- ✅ Dushanbe
|   |   |       |   |   |   |-- ✅ Famagusta
|   |   |       |   |   |   |-- ✅ Gaza
|   |   |       |   |   |   |-- ✅ Harbin
|   |   |       |   |   |   |-- ✅ Hebron
|   |   |       |   |   |   |-- ✅ Ho_Chi_Minh
|   |   |       |   |   |   |-- ✅ Hong_Kong
|   |   |       |   |   |   |-- ✅ Hovd
|   |   |       |   |   |   |-- ✅ Irkutsk
|   |   |       |   |   |   |-- ✅ Istanbul
|   |   |       |   |   |   |-- ✅ Jakarta
|   |   |       |   |   |   |-- ✅ Jayapura
|   |   |       |   |   |   |-- ✅ Jerusalem
|   |   |       |   |   |   |-- ✅ Kabul
|   |   |       |   |   |   |-- ✅ Kamchatka
|   |   |       |   |   |   |-- ✅ Karachi
|   |   |       |   |   |   |-- ✅ Kashgar
|   |   |       |   |   |   |-- ✅ Kathmandu
|   |   |       |   |   |   |-- ✅ Katmandu
|   |   |       |   |   |   |-- ✅ Khandyga
|   |   |       |   |   |   |-- ✅ Kolkata
|   |   |       |   |   |   |-- ✅ Krasnoyarsk
|   |   |       |   |   |   |-- ✅ Kuala_Lumpur
|   |   |       |   |   |   |-- ✅ Kuching
|   |   |       |   |   |   |-- ✅ Kuwait
|   |   |       |   |   |   |-- ✅ Macao
|   |   |       |   |   |   |-- ✅ Macau
|   |   |       |   |   |   |-- ✅ Magadan
|   |   |       |   |   |   |-- ✅ Makassar
|   |   |       |   |   |   |-- ✅ Manila
|   |   |       |   |   |   |-- ✅ Muscat
|   |   |       |   |   |   |-- ✅ Nicosia
|   |   |       |   |   |   |-- ✅ Novokuznetsk
|   |   |       |   |   |   |-- ✅ Novosibirsk
|   |   |       |   |   |   |-- ✅ Omsk
|   |   |       |   |   |   |-- ✅ Oral
|   |   |       |   |   |   |-- ✅ Phnom_Penh
|   |   |       |   |   |   |-- ✅ Pontianak
|   |   |       |   |   |   |-- ✅ Pyongyang
|   |   |       |   |   |   |-- ✅ Qatar
|   |   |       |   |   |   |-- ✅ Qostanay
|   |   |       |   |   |   |-- ✅ Qyzylorda
|   |   |       |   |   |   |-- ✅ Rangoon
|   |   |       |   |   |   |-- ✅ Riyadh
|   |   |       |   |   |   |-- ✅ Saigon
|   |   |       |   |   |   |-- ✅ Sakhalin
|   |   |       |   |   |   |-- ✅ Samarkand
|   |   |       |   |   |   |-- ✅ Seoul
|   |   |       |   |   |   |-- ✅ Shanghai
|   |   |       |   |   |   |-- ✅ Singapore
|   |   |       |   |   |   |-- ✅ Srednekolymsk
|   |   |       |   |   |   |-- ✅ Taipei
|   |   |       |   |   |   |-- ✅ Tashkent
|   |   |       |   |   |   |-- ✅ Tbilisi
|   |   |       |   |   |   |-- ✅ Tehran
|   |   |       |   |   |   |-- ✅ Tel_Aviv
|   |   |       |   |   |   |-- ✅ Thimbu
|   |   |       |   |   |   |-- ✅ Thimphu
|   |   |       |   |   |   |-- ✅ Tokyo
|   |   |       |   |   |   |-- ✅ Tomsk
|   |   |       |   |   |   |-- ✅ Ujung_Pandang
|   |   |       |   |   |   |-- ✅ Ulaanbaatar
|   |   |       |   |   |   |-- ✅ Ulan_Bator
|   |   |       |   |   |   |-- ✅ Urumqi
|   |   |       |   |   |   |-- ✅ Ust-Nera
|   |   |       |   |   |   |-- ✅ Vientiane
|   |   |       |   |   |   |-- ✅ Vladivostok
|   |   |       |   |   |   |-- ✅ Yakutsk
|   |   |       |   |   |   |-- ✅ Yangon
|   |   |       |   |   |   |-- ✅ Yekaterinburg
|   |   |       |   |   |   \-- ✅ Yerevan
|   |   |       |   |   |-- ✅ Atlantic/
|   |   |       |   |   |   |-- ✅ Azores
|   |   |       |   |   |   |-- ✅ Bermuda
|   |   |       |   |   |   |-- ✅ Canary
|   |   |       |   |   |   |-- ✅ Cape_Verde
|   |   |       |   |   |   |-- ✅ Faeroe
|   |   |       |   |   |   |-- ✅ Faroe
|   |   |       |   |   |   |-- ✅ Jan_Mayen
|   |   |       |   |   |   |-- ✅ Madeira
|   |   |       |   |   |   |-- ✅ Reykjavik
|   |   |       |   |   |   |-- ✅ South_Georgia
|   |   |       |   |   |   |-- ✅ St_Helena
|   |   |       |   |   |   \-- ✅ Stanley
|   |   |       |   |   |-- ✅ Australia/
|   |   |       |   |   |   |-- ✅ ACT
|   |   |       |   |   |   |-- ✅ Adelaide
|   |   |       |   |   |   |-- ✅ Brisbane
|   |   |       |   |   |   |-- ✅ Broken_Hill
|   |   |       |   |   |   |-- ✅ Canberra
|   |   |       |   |   |   |-- ✅ Currie
|   |   |       |   |   |   |-- ✅ Darwin
|   |   |       |   |   |   |-- ✅ Eucla
|   |   |       |   |   |   |-- ✅ Hobart
|   |   |       |   |   |   |-- ✅ LHI
|   |   |       |   |   |   |-- ✅ Lindeman
|   |   |       |   |   |   |-- ✅ Lord_Howe
|   |   |       |   |   |   |-- ✅ Melbourne
|   |   |       |   |   |   |-- ✅ North
|   |   |       |   |   |   |-- ✅ NSW
|   |   |       |   |   |   |-- ✅ Perth
|   |   |       |   |   |   |-- ✅ Queensland
|   |   |       |   |   |   |-- ✅ South
|   |   |       |   |   |   |-- ✅ Sydney
|   |   |       |   |   |   |-- ✅ Tasmania
|   |   |       |   |   |   |-- ✅ Victoria
|   |   |       |   |   |   |-- ✅ West
|   |   |       |   |   |   \-- ✅ Yancowinna
|   |   |       |   |   |-- ✅ Brazil/
|   |   |       |   |   |   |-- ✅ Acre
|   |   |       |   |   |   |-- ✅ DeNoronha
|   |   |       |   |   |   |-- ✅ East
|   |   |       |   |   |   \-- ✅ West
|   |   |       |   |   |-- ✅ Canada/
|   |   |       |   |   |   |-- ✅ Atlantic
|   |   |       |   |   |   |-- ✅ Central
|   |   |       |   |   |   |-- ✅ Eastern
|   |   |       |   |   |   |-- ✅ Mountain
|   |   |       |   |   |   |-- ✅ Newfoundland
|   |   |       |   |   |   |-- ✅ Pacific
|   |   |       |   |   |   |-- ✅ Saskatchewan
|   |   |       |   |   |   \-- ✅ Yukon
|   |   |       |   |   |-- ✅ Chile/
|   |   |       |   |   |   |-- ✅ Continental
|   |   |       |   |   |   \-- ✅ EasterIsland
|   |   |       |   |   |-- ✅ Etc/
|   |   |       |   |   |   |-- ✅ GMT
|   |   |       |   |   |   |-- ✅ GMT+0
|   |   |       |   |   |   |-- ✅ GMT+1
|   |   |       |   |   |   |-- ✅ GMT+10
|   |   |       |   |   |   |-- ✅ GMT+11
|   |   |       |   |   |   |-- ✅ GMT+12
|   |   |       |   |   |   |-- ✅ GMT+2
|   |   |       |   |   |   |-- ✅ GMT+3
|   |   |       |   |   |   |-- ✅ GMT+4
|   |   |       |   |   |   |-- ✅ GMT+5
|   |   |       |   |   |   |-- ✅ GMT+6
|   |   |       |   |   |   |-- ✅ GMT+7
|   |   |       |   |   |   |-- ✅ GMT+8
|   |   |       |   |   |   |-- ✅ GMT+9
|   |   |       |   |   |   |-- ✅ GMT-0
|   |   |       |   |   |   |-- ✅ GMT-1
|   |   |       |   |   |   |-- ✅ GMT-10
|   |   |       |   |   |   |-- ✅ GMT-11
|   |   |       |   |   |   |-- ✅ GMT-12
|   |   |       |   |   |   |-- ✅ GMT-13
|   |   |       |   |   |   |-- ✅ GMT-14
|   |   |       |   |   |   |-- ✅ GMT-2
|   |   |       |   |   |   |-- ✅ GMT-3
|   |   |       |   |   |   |-- ✅ GMT-4
|   |   |       |   |   |   |-- ✅ GMT-5
|   |   |       |   |   |   |-- ✅ GMT-6
|   |   |       |   |   |   |-- ✅ GMT-7
|   |   |       |   |   |   |-- ✅ GMT-8
|   |   |       |   |   |   |-- ✅ GMT-9
|   |   |       |   |   |   |-- ✅ GMT0
|   |   |       |   |   |   |-- ✅ Greenwich
|   |   |       |   |   |   |-- ✅ UCT
|   |   |       |   |   |   |-- ✅ Universal
|   |   |       |   |   |   |-- ✅ UTC
|   |   |       |   |   |   \-- ✅ Zulu
|   |   |       |   |   |-- ✅ Europe/
|   |   |       |   |   |   |-- ✅ Amsterdam
|   |   |       |   |   |   |-- ✅ Andorra
|   |   |       |   |   |   |-- ✅ Astrakhan
|   |   |       |   |   |   |-- ✅ Athens
|   |   |       |   |   |   |-- ✅ Belfast
|   |   |       |   |   |   |-- ✅ Belgrade
|   |   |       |   |   |   |-- ✅ Berlin
|   |   |       |   |   |   |-- ✅ Bratislava
|   |   |       |   |   |   |-- ✅ Brussels
|   |   |       |   |   |   |-- ✅ Bucharest
|   |   |       |   |   |   |-- ✅ Budapest
|   |   |       |   |   |   |-- ✅ Busingen
|   |   |       |   |   |   |-- ✅ Chisinau
|   |   |       |   |   |   |-- ✅ Copenhagen
|   |   |       |   |   |   |-- ✅ Dublin
|   |   |       |   |   |   |-- ✅ Gibraltar
|   |   |       |   |   |   |-- ✅ Guernsey
|   |   |       |   |   |   |-- ✅ Helsinki
|   |   |       |   |   |   |-- ✅ Isle_of_Man
|   |   |       |   |   |   |-- ✅ Istanbul
|   |   |       |   |   |   |-- ✅ Jersey
|   |   |       |   |   |   |-- ✅ Kaliningrad
|   |   |       |   |   |   |-- ✅ Kiev
|   |   |       |   |   |   |-- ✅ Kirov
|   |   |       |   |   |   |-- ✅ Kyiv
|   |   |       |   |   |   |-- ✅ Lisbon
|   |   |       |   |   |   |-- ✅ Ljubljana
|   |   |       |   |   |   |-- ✅ London
|   |   |       |   |   |   |-- ✅ Luxembourg
|   |   |       |   |   |   |-- ✅ Madrid
|   |   |       |   |   |   |-- ✅ Malta
|   |   |       |   |   |   |-- ✅ Mariehamn
|   |   |       |   |   |   |-- ✅ Minsk
|   |   |       |   |   |   |-- ✅ Monaco
|   |   |       |   |   |   |-- ✅ Moscow
|   |   |       |   |   |   |-- ✅ Nicosia
|   |   |       |   |   |   |-- ✅ Oslo
|   |   |       |   |   |   |-- ✅ Paris
|   |   |       |   |   |   |-- ✅ Podgorica
|   |   |       |   |   |   |-- ✅ Prague
|   |   |       |   |   |   |-- ✅ Riga
|   |   |       |   |   |   |-- ✅ Rome
|   |   |       |   |   |   |-- ✅ Samara
|   |   |       |   |   |   |-- ✅ San_Marino
|   |   |       |   |   |   |-- ✅ Sarajevo
|   |   |       |   |   |   |-- ✅ Saratov
|   |   |       |   |   |   |-- ✅ Simferopol
|   |   |       |   |   |   |-- ✅ Skopje
|   |   |       |   |   |   |-- ✅ Sofia
|   |   |       |   |   |   |-- ✅ Stockholm
|   |   |       |   |   |   |-- ✅ Tallinn
|   |   |       |   |   |   |-- ✅ Tirane
|   |   |       |   |   |   |-- ✅ Tiraspol
|   |   |       |   |   |   |-- ✅ Ulyanovsk
|   |   |       |   |   |   |-- ✅ Uzhgorod
|   |   |       |   |   |   |-- ✅ Vaduz
|   |   |       |   |   |   |-- ✅ Vatican
|   |   |       |   |   |   |-- ✅ Vienna
|   |   |       |   |   |   |-- ✅ Vilnius
|   |   |       |   |   |   |-- ✅ Volgograd
|   |   |       |   |   |   |-- ✅ Warsaw
|   |   |       |   |   |   |-- ✅ Zagreb
|   |   |       |   |   |   |-- ✅ Zaporozhye
|   |   |       |   |   |   \-- ✅ Zurich
|   |   |       |   |   |-- ✅ Indian/
|   |   |       |   |   |   |-- ✅ Antananarivo
|   |   |       |   |   |   |-- ✅ Chagos
|   |   |       |   |   |   |-- ✅ Christmas
|   |   |       |   |   |   |-- ✅ Cocos
|   |   |       |   |   |   |-- ✅ Comoro
|   |   |       |   |   |   |-- ✅ Kerguelen
|   |   |       |   |   |   |-- ✅ Mahe
|   |   |       |   |   |   |-- ✅ Maldives
|   |   |       |   |   |   |-- ✅ Mauritius
|   |   |       |   |   |   |-- ✅ Mayotte
|   |   |       |   |   |   \-- ✅ Reunion
|   |   |       |   |   |-- ✅ Mexico/
|   |   |       |   |   |   |-- ✅ BajaNorte
|   |   |       |   |   |   |-- ✅ BajaSur
|   |   |       |   |   |   \-- ✅ General
|   |   |       |   |   |-- ✅ Pacific/
|   |   |       |   |   |   |-- ✅ Apia
|   |   |       |   |   |   |-- ✅ Auckland
|   |   |       |   |   |   |-- ✅ Bougainville
|   |   |       |   |   |   |-- ✅ Chatham
|   |   |       |   |   |   |-- ✅ Chuuk
|   |   |       |   |   |   |-- ✅ Easter
|   |   |       |   |   |   |-- ✅ Efate
|   |   |       |   |   |   |-- ✅ Enderbury
|   |   |       |   |   |   |-- ✅ Fakaofo
|   |   |       |   |   |   |-- ✅ Fiji
|   |   |       |   |   |   |-- ✅ Funafuti
|   |   |       |   |   |   |-- ✅ Galapagos
|   |   |       |   |   |   |-- ✅ Gambier
|   |   |       |   |   |   |-- ✅ Guadalcanal
|   |   |       |   |   |   |-- ✅ Guam
|   |   |       |   |   |   |-- ✅ Honolulu
|   |   |       |   |   |   |-- ✅ Johnston
|   |   |       |   |   |   |-- ✅ Kanton
|   |   |       |   |   |   |-- ✅ Kiritimati
|   |   |       |   |   |   |-- ✅ Kosrae
|   |   |       |   |   |   |-- ✅ Kwajalein
|   |   |       |   |   |   |-- ✅ Majuro
|   |   |       |   |   |   |-- ✅ Marquesas
|   |   |       |   |   |   |-- ✅ Midway
|   |   |       |   |   |   |-- ✅ Nauru
|   |   |       |   |   |   |-- ✅ Niue
|   |   |       |   |   |   |-- ✅ Norfolk
|   |   |       |   |   |   |-- ✅ Noumea
|   |   |       |   |   |   |-- ✅ Pago_Pago
|   |   |       |   |   |   |-- ✅ Palau
|   |   |       |   |   |   |-- ✅ Pitcairn
|   |   |       |   |   |   |-- ✅ Pohnpei
|   |   |       |   |   |   |-- ✅ Ponape
|   |   |       |   |   |   |-- ✅ Port_Moresby
|   |   |       |   |   |   |-- ✅ Rarotonga
|   |   |       |   |   |   |-- ✅ Saipan
|   |   |       |   |   |   |-- ✅ Samoa
|   |   |       |   |   |   |-- ✅ Tahiti
|   |   |       |   |   |   |-- ✅ Tarawa
|   |   |       |   |   |   |-- ✅ Tongatapu
|   |   |       |   |   |   |-- ✅ Truk
|   |   |       |   |   |   |-- ✅ Wake
|   |   |       |   |   |   |-- ✅ Wallis
|   |   |       |   |   |   \-- ✅ Yap
|   |   |       |   |   |-- ✅ US/
|   |   |       |   |   |   |-- ✅ Alaska
|   |   |       |   |   |   |-- ✅ Aleutian
|   |   |       |   |   |   |-- ✅ Arizona
|   |   |       |   |   |   |-- ✅ Central
|   |   |       |   |   |   |-- ✅ East-Indiana
|   |   |       |   |   |   |-- ✅ Eastern
|   |   |       |   |   |   |-- ✅ Hawaii
|   |   |       |   |   |   |-- ✅ Indiana-Starke
|   |   |       |   |   |   |-- ✅ Michigan
|   |   |       |   |   |   |-- ✅ Mountain
|   |   |       |   |   |   |-- ✅ Pacific
|   |   |       |   |   |   \-- ✅ Samoa
|   |   |       |   |   |-- ✅ CET
|   |   |       |   |   |-- ✅ CST6CDT
|   |   |       |   |   |-- ✅ Cuba
|   |   |       |   |   |-- ✅ EET
|   |   |       |   |   |-- ✅ Egypt
|   |   |       |   |   |-- ✅ Eire
|   |   |       |   |   |-- ✅ EST
|   |   |       |   |   |-- ✅ EST5EDT
|   |   |       |   |   |-- ✅ Factory
|   |   |       |   |   |-- ✅ GB
|   |   |       |   |   |-- ✅ GB-Eire
|   |   |       |   |   |-- ✅ GMT
|   |   |       |   |   |-- ✅ GMT+0
|   |   |       |   |   |-- ✅ GMT-0
|   |   |       |   |   |-- ✅ GMT0
|   |   |       |   |   |-- ✅ Greenwich
|   |   |       |   |   |-- ✅ Hongkong
|   |   |       |   |   |-- ✅ HST
|   |   |       |   |   |-- ✅ Iceland
|   |   |       |   |   |-- ✅ Iran
|   |   |       |   |   |-- ✅ iso3166.tab
|   |   |       |   |   |-- ✅ Israel
|   |   |       |   |   |-- ✅ Jamaica
|   |   |       |   |   |-- ✅ Japan
|   |   |       |   |   |-- ✅ Kwajalein
|   |   |       |   |   |-- ✅ leapseconds
|   |   |       |   |   |-- ✅ Libya
|   |   |       |   |   |-- ✅ MET
|   |   |       |   |   |-- ✅ MST
|   |   |       |   |   |-- ✅ MST7MDT
|   |   |       |   |   |-- ✅ Navajo
|   |   |       |   |   |-- ✅ NZ
|   |   |       |   |   |-- ✅ NZ-CHAT
|   |   |       |   |   |-- ✅ Poland
|   |   |       |   |   |-- ✅ Portugal
|   |   |       |   |   |-- ✅ PRC
|   |   |       |   |   |-- ✅ PST8PDT
|   |   |       |   |   |-- ✅ ROC
|   |   |       |   |   |-- ✅ ROK
|   |   |       |   |   |-- ✅ Singapore
|   |   |       |   |   |-- ✅ Turkey
|   |   |       |   |   |-- ✅ tzdata.zi
|   |   |       |   |   |-- ✅ UCT
|   |   |       |   |   |-- ✅ Universal
|   |   |       |   |   |-- ✅ UTC
|   |   |       |   |   |-- ✅ W-SU
|   |   |       |   |   |-- ✅ WET
|   |   |       |   |   |-- ✅ zone.tab
|   |   |       |   |   |-- ✅ zone1970.tab
|   |   |       |   |   |-- ✅ zonenow.tab
|   |   |       |   |   \-- ✅ Zulu
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ lazy.py
|   |   |       |   |-- ✅ reference.py
|   |   |       |   |-- ✅ tzfile.py
|   |   |       |   \-- ✅ tzinfo.py
|   |   |       |-- ✅ pytz-2025.2.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   |-- ✅ WHEEL
|   |   |       |   \-- ✅ zip-safe
|   |   |       |-- ✅ regex/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _regex.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _regex_core.py
|   |   |       |   |-- ✅ regex.py
|   |   |       |   \-- ✅ test_regex.py
|   |   |       |-- ✅ regex-2025.9.18.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ requests/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __version__.py
|   |   |       |   |-- ✅ _internal_utils.py
|   |   |       |   |-- ✅ adapters.py
|   |   |       |   |-- ✅ api.py
|   |   |       |   |-- ✅ auth.py
|   |   |       |   |-- ✅ certs.py
|   |   |       |   |-- ✅ compat.py
|   |   |       |   |-- ✅ cookies.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ help.py
|   |   |       |   |-- ✅ hooks.py
|   |   |       |   |-- ✅ models.py
|   |   |       |   |-- ✅ packages.py
|   |   |       |   |-- ✅ sessions.py
|   |   |       |   |-- ✅ status_codes.py
|   |   |       |   |-- ✅ structures.py
|   |   |       |   \-- ✅ utils.py
|   |   |       |-- ✅ requests-2.32.5.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ scikit_learn-1.7.2.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ COPYING
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ scipy/
|   |   |       |   |-- ✅ _lib/
|   |   |       |   |   |-- ✅ _uarray/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _backend.py
|   |   |       |   |   |   |-- ✅ _uarray.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _uarray.cp312-win_amd64.pyd
|   |   |       |   |   |   \-- ✅ LICENSE
|   |   |       |   |   |-- ✅ array_api_compat/
|   |   |       |   |   |   |-- ✅ common/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _aliases.py
|   |   |       |   |   |   |   |-- ✅ _fft.py
|   |   |       |   |   |   |   |-- ✅ _helpers.py
|   |   |       |   |   |   |   |-- ✅ _linalg.py
|   |   |       |   |   |   |   \-- ✅ _typing.py
|   |   |       |   |   |   |-- ✅ cupy/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _aliases.py
|   |   |       |   |   |   |   |-- ✅ _info.py
|   |   |       |   |   |   |   |-- ✅ _typing.py
|   |   |       |   |   |   |   |-- ✅ fft.py
|   |   |       |   |   |   |   \-- ✅ linalg.py
|   |   |       |   |   |   |-- ✅ dask/
|   |   |       |   |   |   |   |-- ✅ array/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ _aliases.py
|   |   |       |   |   |   |   |   |-- ✅ _info.py
|   |   |       |   |   |   |   |   |-- ✅ fft.py
|   |   |       |   |   |   |   |   \-- ✅ linalg.py
|   |   |       |   |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ numpy/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _aliases.py
|   |   |       |   |   |   |   |-- ✅ _info.py
|   |   |       |   |   |   |   |-- ✅ _typing.py
|   |   |       |   |   |   |   |-- ✅ fft.py
|   |   |       |   |   |   |   \-- ✅ linalg.py
|   |   |       |   |   |   |-- ✅ torch/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _aliases.py
|   |   |       |   |   |   |   |-- ✅ _info.py
|   |   |       |   |   |   |   |-- ✅ _typing.py
|   |   |       |   |   |   |   |-- ✅ fft.py
|   |   |       |   |   |   |   \-- ✅ linalg.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ _internal.py
|   |   |       |   |   |-- ✅ array_api_extra/
|   |   |       |   |   |   |-- ✅ _lib/
|   |   |       |   |   |   |   |-- ✅ _utils/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ _compat.py
|   |   |       |   |   |   |   |   |-- ✅ _compat.pyi
|   |   |       |   |   |   |   |   |-- ✅ _helpers.py
|   |   |       |   |   |   |   |   |-- ✅ _typing.py
|   |   |       |   |   |   |   |   \-- ✅ _typing.pyi
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _at.py
|   |   |       |   |   |   |   |-- ✅ _backends.py
|   |   |       |   |   |   |   |-- ✅ _funcs.py
|   |   |       |   |   |   |   |-- ✅ _lazy.py
|   |   |       |   |   |   |   \-- ✅ _testing.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _delegation.py
|   |   |       |   |   |   \-- ✅ testing.py
|   |   |       |   |   |-- ✅ cobyqa/
|   |   |       |   |   |   |-- ✅ subsolvers/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ geometry.py
|   |   |       |   |   |   |   \-- ✅ optim.py
|   |   |       |   |   |   |-- ✅ utils/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ exceptions.py
|   |   |       |   |   |   |   |-- ✅ math.py
|   |   |       |   |   |   |   \-- ✅ versions.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ framework.py
|   |   |       |   |   |   |-- ✅ main.py
|   |   |       |   |   |   |-- ✅ models.py
|   |   |       |   |   |   |-- ✅ problem.py
|   |   |       |   |   |   \-- ✅ settings.py
|   |   |       |   |   |-- ✅ pyprima/
|   |   |       |   |   |   |-- ✅ cobyla/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ cobyla.py
|   |   |       |   |   |   |   |-- ✅ cobylb.py
|   |   |       |   |   |   |   |-- ✅ geometry.py
|   |   |       |   |   |   |   |-- ✅ initialize.py
|   |   |       |   |   |   |   |-- ✅ trustregion.py
|   |   |       |   |   |   |   \-- ✅ update.py
|   |   |       |   |   |   |-- ✅ common/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _bounds.py
|   |   |       |   |   |   |   |-- ✅ _linear_constraints.py
|   |   |       |   |   |   |   |-- ✅ _nonlinear_constraints.py
|   |   |       |   |   |   |   |-- ✅ _project.py
|   |   |       |   |   |   |   |-- ✅ checkbreak.py
|   |   |       |   |   |   |   |-- ✅ consts.py
|   |   |       |   |   |   |   |-- ✅ evaluate.py
|   |   |       |   |   |   |   |-- ✅ history.py
|   |   |       |   |   |   |   |-- ✅ infos.py
|   |   |       |   |   |   |   |-- ✅ linalg.py
|   |   |       |   |   |   |   |-- ✅ message.py
|   |   |       |   |   |   |   |-- ✅ powalg.py
|   |   |       |   |   |   |   |-- ✅ preproc.py
|   |   |       |   |   |   |   |-- ✅ present.py
|   |   |       |   |   |   |   |-- ✅ ratio.py
|   |   |       |   |   |   |   |-- ✅ redrho.py
|   |   |       |   |   |   |   \-- ✅ selectx.py
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test__gcutils.py
|   |   |       |   |   |   |-- ✅ test__pep440.py
|   |   |       |   |   |   |-- ✅ test__testutils.py
|   |   |       |   |   |   |-- ✅ test__threadsafety.py
|   |   |       |   |   |   |-- ✅ test__util.py
|   |   |       |   |   |   |-- ✅ test_array_api.py
|   |   |       |   |   |   |-- ✅ test_bunch.py
|   |   |       |   |   |   |-- ✅ test_ccallback.py
|   |   |       |   |   |   |-- ✅ test_config.py
|   |   |       |   |   |   |-- ✅ test_deprecation.py
|   |   |       |   |   |   |-- ✅ test_doccer.py
|   |   |       |   |   |   |-- ✅ test_import_cycles.py
|   |   |       |   |   |   |-- ✅ test_public_api.py
|   |   |       |   |   |   |-- ✅ test_scipy_version.py
|   |   |       |   |   |   |-- ✅ test_tmpdirs.py
|   |   |       |   |   |   \-- ✅ test_warnings.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _array_api.py
|   |   |       |   |   |-- ✅ _array_api_compat_vendor.py
|   |   |       |   |   |-- ✅ _array_api_no_0d.py
|   |   |       |   |   |-- ✅ _bunch.py
|   |   |       |   |   |-- ✅ _ccallback.py
|   |   |       |   |   |-- ✅ _ccallback_c.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _ccallback_c.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _disjoint_set.py
|   |   |       |   |   |-- ✅ _docscrape.py
|   |   |       |   |   |-- ✅ _elementwise_iterative_method.py
|   |   |       |   |   |-- ✅ _fpumode.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _fpumode.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _gcutils.py
|   |   |       |   |   |-- ✅ _pep440.py
|   |   |       |   |   |-- ✅ _sparse.py
|   |   |       |   |   |-- ✅ _test_ccallback.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _test_ccallback.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _test_deprecation_call.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _test_deprecation_call.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _test_deprecation_def.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _test_deprecation_def.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _testutils.py
|   |   |       |   |   |-- ✅ _threadsafety.py
|   |   |       |   |   |-- ✅ _tmpdirs.py
|   |   |       |   |   |-- ✅ _util.py
|   |   |       |   |   |-- ✅ decorator.py
|   |   |       |   |   |-- ✅ deprecation.py
|   |   |       |   |   |-- ✅ doccer.py
|   |   |       |   |   |-- ✅ messagestream.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ messagestream.cp312-win_amd64.pyd
|   |   |       |   |   \-- ✅ uarray.py
|   |   |       |   |-- ✅ cluster/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ hierarchy_test_data.py
|   |   |       |   |   |   |-- ✅ test_disjoint_set.py
|   |   |       |   |   |   |-- ✅ test_hierarchy.py
|   |   |       |   |   |   \-- ✅ test_vq.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _hierarchy.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _hierarchy.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _optimal_leaf_ordering.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _optimal_leaf_ordering.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _vq.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _vq.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ hierarchy.py
|   |   |       |   |   \-- ✅ vq.py
|   |   |       |   |-- ✅ constants/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_codata.py
|   |   |       |   |   |   \-- ✅ test_constants.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _codata.py
|   |   |       |   |   |-- ✅ _constants.py
|   |   |       |   |   |-- ✅ codata.py
|   |   |       |   |   \-- ✅ constants.py
|   |   |       |   |-- ✅ datasets/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ test_data.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _download_all.py
|   |   |       |   |   |-- ✅ _fetchers.py
|   |   |       |   |   |-- ✅ _registry.py
|   |   |       |   |   \-- ✅ _utils.py
|   |   |       |   |-- ✅ differentiate/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ test_differentiate.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ _differentiate.py
|   |   |       |   |-- ✅ fft/
|   |   |       |   |   |-- ✅ _pocketfft/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_basic.py
|   |   |       |   |   |   |   \-- ✅ test_real_transforms.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ basic.py
|   |   |       |   |   |   |-- ✅ helper.py
|   |   |       |   |   |   |-- ✅ LICENSE.md
|   |   |       |   |   |   |-- ✅ pypocketfft.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ pypocketfft.cp312-win_amd64.pyd
|   |   |       |   |   |   \-- ✅ realtransforms.py
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ mock_backend.py
|   |   |       |   |   |   |-- ✅ test_backend.py
|   |   |       |   |   |   |-- ✅ test_basic.py
|   |   |       |   |   |   |-- ✅ test_fftlog.py
|   |   |       |   |   |   |-- ✅ test_helper.py
|   |   |       |   |   |   |-- ✅ test_multithreading.py
|   |   |       |   |   |   \-- ✅ test_real_transforms.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _backend.py
|   |   |       |   |   |-- ✅ _basic.py
|   |   |       |   |   |-- ✅ _basic_backend.py
|   |   |       |   |   |-- ✅ _debug_backends.py
|   |   |       |   |   |-- ✅ _fftlog.py
|   |   |       |   |   |-- ✅ _fftlog_backend.py
|   |   |       |   |   |-- ✅ _helper.py
|   |   |       |   |   |-- ✅ _realtransforms.py
|   |   |       |   |   \-- ✅ _realtransforms_backend.py
|   |   |       |   |-- ✅ fftpack/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ fftw_double_ref.npz
|   |   |       |   |   |   |-- ✅ fftw_longdouble_ref.npz
|   |   |       |   |   |   |-- ✅ fftw_single_ref.npz
|   |   |       |   |   |   |-- ✅ test.npz
|   |   |       |   |   |   |-- ✅ test_basic.py
|   |   |       |   |   |   |-- ✅ test_helper.py
|   |   |       |   |   |   |-- ✅ test_import.py
|   |   |       |   |   |   |-- ✅ test_pseudo_diffs.py
|   |   |       |   |   |   \-- ✅ test_real_transforms.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _basic.py
|   |   |       |   |   |-- ✅ _helper.py
|   |   |       |   |   |-- ✅ _pseudo_diffs.py
|   |   |       |   |   |-- ✅ _realtransforms.py
|   |   |       |   |   |-- ✅ basic.py
|   |   |       |   |   |-- ✅ convolve.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ convolve.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ helper.py
|   |   |       |   |   |-- ✅ pseudo_diffs.py
|   |   |       |   |   \-- ✅ realtransforms.py
|   |   |       |   |-- ✅ integrate/
|   |   |       |   |   |-- ✅ _ivp/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_ivp.py
|   |   |       |   |   |   |   \-- ✅ test_rk.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ bdf.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ dop853_coefficients.py
|   |   |       |   |   |   |-- ✅ ivp.py
|   |   |       |   |   |   |-- ✅ lsoda.py
|   |   |       |   |   |   |-- ✅ radau.py
|   |   |       |   |   |   \-- ✅ rk.py
|   |   |       |   |   |-- ✅ _rules/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _base.py
|   |   |       |   |   |   |-- ✅ _gauss_kronrod.py
|   |   |       |   |   |   |-- ✅ _gauss_legendre.py
|   |   |       |   |   |   \-- ✅ _genz_malik.py
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test__quad_vec.py
|   |   |       |   |   |   |-- ✅ test_banded_ode_solvers.py
|   |   |       |   |   |   |-- ✅ test_bvp.py
|   |   |       |   |   |   |-- ✅ test_cubature.py
|   |   |       |   |   |   |-- ✅ test_integrate.py
|   |   |       |   |   |   |-- ✅ test_odeint_jac.py
|   |   |       |   |   |   |-- ✅ test_quadpack.py
|   |   |       |   |   |   |-- ✅ test_quadrature.py
|   |   |       |   |   |   \-- ✅ test_tanhsinh.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _bvp.py
|   |   |       |   |   |-- ✅ _cubature.py
|   |   |       |   |   |-- ✅ _dop.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _dop.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _lebedev.py
|   |   |       |   |   |-- ✅ _lsoda.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _lsoda.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _ode.py
|   |   |       |   |   |-- ✅ _odepack.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _odepack.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _odepack_py.py
|   |   |       |   |   |-- ✅ _quad_vec.py
|   |   |       |   |   |-- ✅ _quadpack.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _quadpack.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _quadpack_py.py
|   |   |       |   |   |-- ✅ _quadrature.py
|   |   |       |   |   |-- ✅ _tanhsinh.py
|   |   |       |   |   |-- ✅ _test_multivariate.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _test_multivariate.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _test_odeint_banded.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _test_odeint_banded.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _vode.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _vode.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ dop.py
|   |   |       |   |   |-- ✅ lsoda.py
|   |   |       |   |   |-- ✅ odepack.py
|   |   |       |   |   |-- ✅ quadpack.py
|   |   |       |   |   \-- ✅ vode.py
|   |   |       |   |-- ✅ interpolate/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ bug-1310.npz
|   |   |       |   |   |   |   |-- ✅ estimate_gradients_hang.npy
|   |   |       |   |   |   |   \-- ✅ gcvspl.npz
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_bary_rational.py
|   |   |       |   |   |   |-- ✅ test_bsplines.py
|   |   |       |   |   |   |-- ✅ test_fitpack.py
|   |   |       |   |   |   |-- ✅ test_fitpack2.py
|   |   |       |   |   |   |-- ✅ test_gil.py
|   |   |       |   |   |   |-- ✅ test_interpnd.py
|   |   |       |   |   |   |-- ✅ test_interpolate.py
|   |   |       |   |   |   |-- ✅ test_ndgriddata.py
|   |   |       |   |   |   |-- ✅ test_pade.py
|   |   |       |   |   |   |-- ✅ test_polyint.py
|   |   |       |   |   |   |-- ✅ test_rbf.py
|   |   |       |   |   |   |-- ✅ test_rbfinterp.py
|   |   |       |   |   |   \-- ✅ test_rgi.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _bary_rational.py
|   |   |       |   |   |-- ✅ _bsplines.py
|   |   |       |   |   |-- ✅ _cubic.py
|   |   |       |   |   |-- ✅ _dfitpack.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _dfitpack.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _dierckx.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _dierckx.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _fitpack.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _fitpack.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _fitpack2.py
|   |   |       |   |   |-- ✅ _fitpack_impl.py
|   |   |       |   |   |-- ✅ _fitpack_py.py
|   |   |       |   |   |-- ✅ _fitpack_repro.py
|   |   |       |   |   |-- ✅ _interpnd.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _interpnd.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _interpolate.py
|   |   |       |   |   |-- ✅ _ndbspline.py
|   |   |       |   |   |-- ✅ _ndgriddata.py
|   |   |       |   |   |-- ✅ _pade.py
|   |   |       |   |   |-- ✅ _polyint.py
|   |   |       |   |   |-- ✅ _ppoly.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _ppoly.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _rbf.py
|   |   |       |   |   |-- ✅ _rbfinterp.py
|   |   |       |   |   |-- ✅ _rbfinterp_pythran.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _rbfinterp_pythran.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _rgi.py
|   |   |       |   |   |-- ✅ _rgi_cython.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _rgi_cython.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ dfitpack.py
|   |   |       |   |   |-- ✅ fitpack.py
|   |   |       |   |   |-- ✅ fitpack2.py
|   |   |       |   |   |-- ✅ interpnd.py
|   |   |       |   |   |-- ✅ interpolate.py
|   |   |       |   |   |-- ✅ ndgriddata.py
|   |   |       |   |   |-- ✅ polyint.py
|   |   |       |   |   \-- ✅ rbf.py
|   |   |       |   |-- ✅ io/
|   |   |       |   |   |-- ✅ _fast_matrix_market/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _fmm_core.cp312-win_amd64.dll.a
|   |   |       |   |   |   \-- ✅ _fmm_core.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _harwell_boeing/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_fortran_format.py
|   |   |       |   |   |   |   \-- ✅ test_hb.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _fortran_format_parser.py
|   |   |       |   |   |   \-- ✅ hb.py
|   |   |       |   |   |-- ✅ arff/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |   |-- ✅ iris.arff
|   |   |       |   |   |   |   |   |-- ✅ missing.arff
|   |   |       |   |   |   |   |   |-- ✅ nodata.arff
|   |   |       |   |   |   |   |   |-- ✅ quoted_nominal.arff
|   |   |       |   |   |   |   |   |-- ✅ quoted_nominal_spaces.arff
|   |   |       |   |   |   |   |   |-- ✅ test1.arff
|   |   |       |   |   |   |   |   |-- ✅ test10.arff
|   |   |       |   |   |   |   |   |-- ✅ test11.arff
|   |   |       |   |   |   |   |   |-- ✅ test2.arff
|   |   |       |   |   |   |   |   |-- ✅ test3.arff
|   |   |       |   |   |   |   |   |-- ✅ test4.arff
|   |   |       |   |   |   |   |   |-- ✅ test5.arff
|   |   |       |   |   |   |   |   |-- ✅ test6.arff
|   |   |       |   |   |   |   |   |-- ✅ test7.arff
|   |   |       |   |   |   |   |   |-- ✅ test8.arff
|   |   |       |   |   |   |   |   \-- ✅ test9.arff
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ test_arffread.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _arffread.py
|   |   |       |   |   |   \-- ✅ arffread.py
|   |   |       |   |   |-- ✅ matlab/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |   |-- ✅ bad_miuint32.mat
|   |   |       |   |   |   |   |   |-- ✅ bad_miutf8_array_name.mat
|   |   |       |   |   |   |   |   |-- ✅ big_endian.mat
|   |   |       |   |   |   |   |   |-- ✅ broken_utf8.mat
|   |   |       |   |   |   |   |   |-- ✅ corrupted_zlib_checksum.mat
|   |   |       |   |   |   |   |   |-- ✅ corrupted_zlib_data.mat
|   |   |       |   |   |   |   |   |-- ✅ debigged_m4.mat
|   |   |       |   |   |   |   |   |-- ✅ japanese_utf8.txt
|   |   |       |   |   |   |   |   |-- ✅ little_endian.mat
|   |   |       |   |   |   |   |   |-- ✅ logical_sparse.mat
|   |   |       |   |   |   |   |   |-- ✅ malformed1.mat
|   |   |       |   |   |   |   |   |-- ✅ miuint32_for_miint32.mat
|   |   |       |   |   |   |   |   |-- ✅ miutf8_array_name.mat
|   |   |       |   |   |   |   |   |-- ✅ nasty_duplicate_fieldnames.mat
|   |   |       |   |   |   |   |   |-- ✅ one_by_zero_char.mat
|   |   |       |   |   |   |   |   |-- ✅ parabola.mat
|   |   |       |   |   |   |   |   |-- ✅ single_empty_string.mat
|   |   |       |   |   |   |   |   |-- ✅ some_functions.mat
|   |   |       |   |   |   |   |   |-- ✅ sqr.mat
|   |   |       |   |   |   |   |   |-- ✅ test3dmatrix_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ test3dmatrix_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ test3dmatrix_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ test3dmatrix_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ test_empty_struct.mat
|   |   |       |   |   |   |   |   |-- ✅ test_mat4_le_floats.mat
|   |   |       |   |   |   |   |   |-- ✅ test_skip_variable.mat
|   |   |       |   |   |   |   |   |-- ✅ testbool_8_WIN64.mat
|   |   |       |   |   |   |   |   |-- ✅ testcell_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testcell_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testcell_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testcell_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testcellnest_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testcellnest_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testcellnest_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testcellnest_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testcomplex_4.2c_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testcomplex_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testcomplex_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testcomplex_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testcomplex_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testdouble_4.2c_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testdouble_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testdouble_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testdouble_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testdouble_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testemptycell_5.3_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testemptycell_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testemptycell_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testemptycell_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testfunc_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testhdf5_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testmatrix_4.2c_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testmatrix_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testmatrix_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testmatrix_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testmatrix_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testminus_4.2c_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testminus_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testminus_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testminus_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testminus_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testmulti_4.2c_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testmulti_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testmulti_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testobject_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testobject_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testobject_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testobject_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testonechar_4.2c_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testonechar_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testonechar_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testonechar_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testonechar_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testscalarcell_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testsimplecell.mat
|   |   |       |   |   |   |   |   |-- ✅ testsparse_4.2c_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testsparse_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testsparse_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testsparse_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testsparse_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testsparsecomplex_4.2c_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testsparsecomplex_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ testsparsecomplex_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testsparsecomplex_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testsparsecomplex_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testsparsefloat_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststring_4.2c_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ teststring_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ teststring_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststring_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststring_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststringarray_4.2c_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ teststringarray_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ teststringarray_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststringarray_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststringarray_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststruct_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ teststruct_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststruct_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststruct_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststructarr_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ teststructarr_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststructarr_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststructarr_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststructnest_6.1_SOL2.mat
|   |   |       |   |   |   |   |   |-- ✅ teststructnest_6.5.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststructnest_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ teststructnest_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testunicode_7.1_GLNX86.mat
|   |   |       |   |   |   |   |   |-- ✅ testunicode_7.4_GLNX86.mat
|   |   |       |   |   |   |   |   \-- ✅ testvec_4_GLNX86.mat
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_byteordercodes.py
|   |   |       |   |   |   |   |-- ✅ test_mio.py
|   |   |       |   |   |   |   |-- ✅ test_mio5_utils.py
|   |   |       |   |   |   |   |-- ✅ test_mio_funcs.py
|   |   |       |   |   |   |   |-- ✅ test_mio_utils.py
|   |   |       |   |   |   |   |-- ✅ test_miobase.py
|   |   |       |   |   |   |   |-- ✅ test_pathological.py
|   |   |       |   |   |   |   \-- ✅ test_streams.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _byteordercodes.py
|   |   |       |   |   |   |-- ✅ _mio.py
|   |   |       |   |   |   |-- ✅ _mio4.py
|   |   |       |   |   |   |-- ✅ _mio5.py
|   |   |       |   |   |   |-- ✅ _mio5_params.py
|   |   |       |   |   |   |-- ✅ _mio5_utils.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _mio5_utils.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _mio_utils.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _mio_utils.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _miobase.py
|   |   |       |   |   |   |-- ✅ _streams.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _streams.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ byteordercodes.py
|   |   |       |   |   |   |-- ✅ mio.py
|   |   |       |   |   |   |-- ✅ mio4.py
|   |   |       |   |   |   |-- ✅ mio5.py
|   |   |       |   |   |   |-- ✅ mio5_params.py
|   |   |       |   |   |   |-- ✅ mio5_utils.py
|   |   |       |   |   |   |-- ✅ mio_utils.py
|   |   |       |   |   |   |-- ✅ miobase.py
|   |   |       |   |   |   \-- ✅ streams.py
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ array_float32_1d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_2d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_3d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_4d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_5d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_6d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_7d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_8d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_pointer_1d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_pointer_2d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_pointer_3d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_pointer_4d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_pointer_5d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_pointer_6d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_pointer_7d.sav
|   |   |       |   |   |   |   |-- ✅ array_float32_pointer_8d.sav
|   |   |       |   |   |   |   |-- ✅ example_1.nc
|   |   |       |   |   |   |   |-- ✅ example_2.nc
|   |   |       |   |   |   |   |-- ✅ example_3_maskedvals.nc
|   |   |       |   |   |   |   |-- ✅ fortran-3x3d-2i.dat
|   |   |       |   |   |   |   |-- ✅ fortran-mixed.dat
|   |   |       |   |   |   |   |-- ✅ fortran-sf8-11x1x10.dat
|   |   |       |   |   |   |   |-- ✅ fortran-sf8-15x10x22.dat
|   |   |       |   |   |   |   |-- ✅ fortran-sf8-1x1x1.dat
|   |   |       |   |   |   |   |-- ✅ fortran-sf8-1x1x5.dat
|   |   |       |   |   |   |   |-- ✅ fortran-sf8-1x1x7.dat
|   |   |       |   |   |   |   |-- ✅ fortran-sf8-1x3x5.dat
|   |   |       |   |   |   |   |-- ✅ fortran-si4-11x1x10.dat
|   |   |       |   |   |   |   |-- ✅ fortran-si4-15x10x22.dat
|   |   |       |   |   |   |   |-- ✅ fortran-si4-1x1x1.dat
|   |   |       |   |   |   |   |-- ✅ fortran-si4-1x1x5.dat
|   |   |       |   |   |   |   |-- ✅ fortran-si4-1x1x7.dat
|   |   |       |   |   |   |   |-- ✅ fortran-si4-1x3x5.dat
|   |   |       |   |   |   |   |-- ✅ invalid_pointer.sav
|   |   |       |   |   |   |   |-- ✅ null_pointer.sav
|   |   |       |   |   |   |   |-- ✅ scalar_byte.sav
|   |   |       |   |   |   |   |-- ✅ scalar_byte_descr.sav
|   |   |       |   |   |   |   |-- ✅ scalar_complex32.sav
|   |   |       |   |   |   |   |-- ✅ scalar_complex64.sav
|   |   |       |   |   |   |   |-- ✅ scalar_float32.sav
|   |   |       |   |   |   |   |-- ✅ scalar_float64.sav
|   |   |       |   |   |   |   |-- ✅ scalar_heap_pointer.sav
|   |   |       |   |   |   |   |-- ✅ scalar_int16.sav
|   |   |       |   |   |   |   |-- ✅ scalar_int32.sav
|   |   |       |   |   |   |   |-- ✅ scalar_int64.sav
|   |   |       |   |   |   |   |-- ✅ scalar_string.sav
|   |   |       |   |   |   |   |-- ✅ scalar_uint16.sav
|   |   |       |   |   |   |   |-- ✅ scalar_uint32.sav
|   |   |       |   |   |   |   |-- ✅ scalar_uint64.sav
|   |   |       |   |   |   |   |-- ✅ struct_arrays.sav
|   |   |       |   |   |   |   |-- ✅ struct_arrays_byte_idl80.sav
|   |   |       |   |   |   |   |-- ✅ struct_arrays_replicated.sav
|   |   |       |   |   |   |   |-- ✅ struct_arrays_replicated_3d.sav
|   |   |       |   |   |   |   |-- ✅ struct_inherit.sav
|   |   |       |   |   |   |   |-- ✅ struct_pointer_arrays.sav
|   |   |       |   |   |   |   |-- ✅ struct_pointer_arrays_replicated.sav
|   |   |       |   |   |   |   |-- ✅ struct_pointer_arrays_replicated_3d.sav
|   |   |       |   |   |   |   |-- ✅ struct_pointers.sav
|   |   |       |   |   |   |   |-- ✅ struct_pointers_replicated.sav
|   |   |       |   |   |   |   |-- ✅ struct_pointers_replicated_3d.sav
|   |   |       |   |   |   |   |-- ✅ struct_scalars.sav
|   |   |       |   |   |   |   |-- ✅ struct_scalars_replicated.sav
|   |   |       |   |   |   |   |-- ✅ struct_scalars_replicated_3d.sav
|   |   |       |   |   |   |   |-- ✅ test-1234Hz-le-1ch-10S-20bit-extra.wav
|   |   |       |   |   |   |   |-- ✅ test-44100Hz-2ch-32bit-float-be.wav
|   |   |       |   |   |   |   |-- ✅ test-44100Hz-2ch-32bit-float-le.wav
|   |   |       |   |   |   |   |-- ✅ test-44100Hz-be-1ch-4bytes.wav
|   |   |       |   |   |   |   |-- ✅ test-44100Hz-le-1ch-4bytes-early-eof-no-data.wav
|   |   |       |   |   |   |   |-- ✅ test-44100Hz-le-1ch-4bytes-early-eof.wav
|   |   |       |   |   |   |   |-- ✅ test-44100Hz-le-1ch-4bytes-incomplete-chunk.wav
|   |   |       |   |   |   |   |-- ✅ test-44100Hz-le-1ch-4bytes-rf64.wav
|   |   |       |   |   |   |   |-- ✅ test-44100Hz-le-1ch-4bytes.wav
|   |   |       |   |   |   |   |-- ✅ test-48000Hz-2ch-64bit-float-le-wavex.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-be-3ch-5S-24bit.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-le-1ch-1byte-ulaw.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-le-2ch-1byteu.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-le-3ch-5S-24bit-inconsistent.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-le-3ch-5S-24bit-rf64.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-le-3ch-5S-24bit.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-le-3ch-5S-36bit.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-le-3ch-5S-45bit.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-le-3ch-5S-53bit.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-le-3ch-5S-64bit.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-le-4ch-9S-12bit.wav
|   |   |       |   |   |   |   |-- ✅ test-8000Hz-le-5ch-9S-5bit.wav
|   |   |       |   |   |   |   |-- ✅ Transparent Busy.ani
|   |   |       |   |   |   |   \-- ✅ various_compressed.sav
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_fortran.py
|   |   |       |   |   |   |-- ✅ test_idl.py
|   |   |       |   |   |   |-- ✅ test_mmio.py
|   |   |       |   |   |   |-- ✅ test_netcdf.py
|   |   |       |   |   |   |-- ✅ test_paths.py
|   |   |       |   |   |   \-- ✅ test_wavfile.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _fortran.py
|   |   |       |   |   |-- ✅ _idl.py
|   |   |       |   |   |-- ✅ _mmio.py
|   |   |       |   |   |-- ✅ _netcdf.py
|   |   |       |   |   |-- ✅ _test_fortran.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _test_fortran.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ harwell_boeing.py
|   |   |       |   |   |-- ✅ idl.py
|   |   |       |   |   |-- ✅ mmio.py
|   |   |       |   |   |-- ✅ netcdf.py
|   |   |       |   |   \-- ✅ wavfile.py
|   |   |       |   |-- ✅ linalg/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ _cython_examples/
|   |   |       |   |   |   |   |-- ✅ extending.pyx
|   |   |       |   |   |   |   \-- ✅ meson.build
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ carex_15_data.npz
|   |   |       |   |   |   |   |-- ✅ carex_18_data.npz
|   |   |       |   |   |   |   |-- ✅ carex_19_data.npz
|   |   |       |   |   |   |   |-- ✅ carex_20_data.npz
|   |   |       |   |   |   |   |-- ✅ carex_6_data.npz
|   |   |       |   |   |   |   \-- ✅ gendare_20170120_data.npz
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_basic.py
|   |   |       |   |   |   |-- ✅ test_batch.py
|   |   |       |   |   |   |-- ✅ test_blas.py
|   |   |       |   |   |   |-- ✅ test_cython_blas.py
|   |   |       |   |   |   |-- ✅ test_cython_lapack.py
|   |   |       |   |   |   |-- ✅ test_cythonized_array_utils.py
|   |   |       |   |   |   |-- ✅ test_decomp.py
|   |   |       |   |   |   |-- ✅ test_decomp_cholesky.py
|   |   |       |   |   |   |-- ✅ test_decomp_cossin.py
|   |   |       |   |   |   |-- ✅ test_decomp_ldl.py
|   |   |       |   |   |   |-- ✅ test_decomp_lu.py
|   |   |       |   |   |   |-- ✅ test_decomp_polar.py
|   |   |       |   |   |   |-- ✅ test_decomp_update.py
|   |   |       |   |   |   |-- ✅ test_extending.py
|   |   |       |   |   |   |-- ✅ test_fblas.py
|   |   |       |   |   |   |-- ✅ test_interpolative.py
|   |   |       |   |   |   |-- ✅ test_lapack.py
|   |   |       |   |   |   |-- ✅ test_matfuncs.py
|   |   |       |   |   |   |-- ✅ test_matmul_toeplitz.py
|   |   |       |   |   |   |-- ✅ test_procrustes.py
|   |   |       |   |   |   |-- ✅ test_sketches.py
|   |   |       |   |   |   |-- ✅ test_solve_toeplitz.py
|   |   |       |   |   |   |-- ✅ test_solvers.py
|   |   |       |   |   |   \-- ✅ test_special_matrices.py
|   |   |       |   |   |-- ✅ __init__.pxd
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _basic.py
|   |   |       |   |   |-- ✅ _blas_subroutines.h
|   |   |       |   |   |-- ✅ _cythonized_array_utils.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _cythonized_array_utils.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _cythonized_array_utils.pxd
|   |   |       |   |   |-- ✅ _cythonized_array_utils.pyi
|   |   |       |   |   |-- ✅ _decomp.py
|   |   |       |   |   |-- ✅ _decomp_cholesky.py
|   |   |       |   |   |-- ✅ _decomp_cossin.py
|   |   |       |   |   |-- ✅ _decomp_interpolative.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _decomp_interpolative.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _decomp_ldl.py
|   |   |       |   |   |-- ✅ _decomp_lu.py
|   |   |       |   |   |-- ✅ _decomp_lu_cython.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _decomp_lu_cython.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _decomp_lu_cython.pyi
|   |   |       |   |   |-- ✅ _decomp_polar.py
|   |   |       |   |   |-- ✅ _decomp_qr.py
|   |   |       |   |   |-- ✅ _decomp_qz.py
|   |   |       |   |   |-- ✅ _decomp_schur.py
|   |   |       |   |   |-- ✅ _decomp_svd.py
|   |   |       |   |   |-- ✅ _decomp_update.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _decomp_update.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _expm_frechet.py
|   |   |       |   |   |-- ✅ _fblas.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _fblas.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _flapack.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _flapack.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _lapack_subroutines.h
|   |   |       |   |   |-- ✅ _linalg_pythran.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _linalg_pythran.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _matfuncs.py
|   |   |       |   |   |-- ✅ _matfuncs_expm.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _matfuncs_expm.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _matfuncs_expm.pyi
|   |   |       |   |   |-- ✅ _matfuncs_inv_ssq.py
|   |   |       |   |   |-- ✅ _matfuncs_schur_sqrtm.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _matfuncs_schur_sqrtm.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _matfuncs_sqrtm.py
|   |   |       |   |   |-- ✅ _matfuncs_sqrtm_triu.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _matfuncs_sqrtm_triu.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _misc.py
|   |   |       |   |   |-- ✅ _procrustes.py
|   |   |       |   |   |-- ✅ _sketches.py
|   |   |       |   |   |-- ✅ _solve_toeplitz.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _solve_toeplitz.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _solvers.py
|   |   |       |   |   |-- ✅ _special_matrices.py
|   |   |       |   |   |-- ✅ _testutils.py
|   |   |       |   |   |-- ✅ basic.py
|   |   |       |   |   |-- ✅ blas.py
|   |   |       |   |   |-- ✅ cython_blas.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ cython_blas.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ cython_blas.pxd
|   |   |       |   |   |-- ✅ cython_blas.pyx
|   |   |       |   |   |-- ✅ cython_lapack.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ cython_lapack.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ cython_lapack.pxd
|   |   |       |   |   |-- ✅ cython_lapack.pyx
|   |   |       |   |   |-- ✅ decomp.py
|   |   |       |   |   |-- ✅ decomp_cholesky.py
|   |   |       |   |   |-- ✅ decomp_lu.py
|   |   |       |   |   |-- ✅ decomp_qr.py
|   |   |       |   |   |-- ✅ decomp_schur.py
|   |   |       |   |   |-- ✅ decomp_svd.py
|   |   |       |   |   |-- ✅ interpolative.py
|   |   |       |   |   |-- ✅ lapack.py
|   |   |       |   |   |-- ✅ matfuncs.py
|   |   |       |   |   |-- ✅ misc.py
|   |   |       |   |   \-- ✅ special_matrices.py
|   |   |       |   |-- ✅ misc/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ common.py
|   |   |       |   |   \-- ✅ doccer.py
|   |   |       |   |-- ✅ ndimage/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ label_inputs.txt
|   |   |       |   |   |   |   |-- ✅ label_results.txt
|   |   |       |   |   |   |   \-- ✅ label_strels.txt
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ dots.png
|   |   |       |   |   |   |-- ✅ test_c_api.py
|   |   |       |   |   |   |-- ✅ test_datatypes.py
|   |   |       |   |   |   |-- ✅ test_filters.py
|   |   |       |   |   |   |-- ✅ test_fourier.py
|   |   |       |   |   |   |-- ✅ test_interpolation.py
|   |   |       |   |   |   |-- ✅ test_measurements.py
|   |   |       |   |   |   |-- ✅ test_morphology.py
|   |   |       |   |   |   |-- ✅ test_ni_support.py
|   |   |       |   |   |   \-- ✅ test_splines.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _ctest.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _ctest.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _cytest.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _cytest.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _delegators.py
|   |   |       |   |   |-- ✅ _filters.py
|   |   |       |   |   |-- ✅ _fourier.py
|   |   |       |   |   |-- ✅ _interpolation.py
|   |   |       |   |   |-- ✅ _measurements.py
|   |   |       |   |   |-- ✅ _morphology.py
|   |   |       |   |   |-- ✅ _nd_image.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _nd_image.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _ndimage_api.py
|   |   |       |   |   |-- ✅ _ni_docstrings.py
|   |   |       |   |   |-- ✅ _ni_label.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _ni_label.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _ni_support.py
|   |   |       |   |   |-- ✅ _rank_filter_1d.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _rank_filter_1d.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _support_alternative_backends.py
|   |   |       |   |   |-- ✅ filters.py
|   |   |       |   |   |-- ✅ fourier.py
|   |   |       |   |   |-- ✅ interpolation.py
|   |   |       |   |   |-- ✅ measurements.py
|   |   |       |   |   \-- ✅ morphology.py
|   |   |       |   |-- ✅ odr/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ test_odr.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ __odrpack.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ __odrpack.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _add_newdocs.py
|   |   |       |   |   |-- ✅ _models.py
|   |   |       |   |   |-- ✅ _odrpack.py
|   |   |       |   |   |-- ✅ models.py
|   |   |       |   |   \-- ✅ odrpack.py
|   |   |       |   |-- ✅ optimize/
|   |   |       |   |   |-- ✅ _highspy/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _core.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _core.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _highs_options.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _highs_options.cp312-win_amd64.pyd
|   |   |       |   |   |   \-- ✅ _highs_wrapper.py
|   |   |       |   |   |-- ✅ _lsq/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ bvls.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ dogbox.py
|   |   |       |   |   |   |-- ✅ givens_elimination.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ givens_elimination.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ least_squares.py
|   |   |       |   |   |   |-- ✅ lsq_linear.py
|   |   |       |   |   |   |-- ✅ trf.py
|   |   |       |   |   |   \-- ✅ trf_linear.py
|   |   |       |   |   |-- ✅ _shgo_lib/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _complex.py
|   |   |       |   |   |   \-- ✅ _vertex.py
|   |   |       |   |   |-- ✅ _trlib/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _trlib.cp312-win_amd64.dll.a
|   |   |       |   |   |   \-- ✅ _trlib.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _trustregion_constr/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_canonical_constraint.py
|   |   |       |   |   |   |   |-- ✅ test_nested_minimize.py
|   |   |       |   |   |   |   |-- ✅ test_projections.py
|   |   |       |   |   |   |   |-- ✅ test_qp_subproblem.py
|   |   |       |   |   |   |   \-- ✅ test_report.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ canonical_constraint.py
|   |   |       |   |   |   |-- ✅ equality_constrained_sqp.py
|   |   |       |   |   |   |-- ✅ minimize_trustregion_constr.py
|   |   |       |   |   |   |-- ✅ projections.py
|   |   |       |   |   |   |-- ✅ qp_subproblem.py
|   |   |       |   |   |   |-- ✅ report.py
|   |   |       |   |   |   \-- ✅ tr_interior_point.py
|   |   |       |   |   |-- ✅ cython_optimize/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _zeros.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _zeros.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _zeros.pxd
|   |   |       |   |   |   \-- ✅ c_zeros.pxd
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ _cython_examples/
|   |   |       |   |   |   |   |-- ✅ extending.pyx
|   |   |       |   |   |   |   \-- ✅ meson.build
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test__basinhopping.py
|   |   |       |   |   |   |-- ✅ test__differential_evolution.py
|   |   |       |   |   |   |-- ✅ test__dual_annealing.py
|   |   |       |   |   |   |-- ✅ test__linprog_clean_inputs.py
|   |   |       |   |   |   |-- ✅ test__numdiff.py
|   |   |       |   |   |   |-- ✅ test__remove_redundancy.py
|   |   |       |   |   |   |-- ✅ test__root.py
|   |   |       |   |   |   |-- ✅ test__shgo.py
|   |   |       |   |   |   |-- ✅ test__spectral.py
|   |   |       |   |   |   |-- ✅ test_bracket.py
|   |   |       |   |   |   |-- ✅ test_chandrupatla.py
|   |   |       |   |   |   |-- ✅ test_cobyla.py
|   |   |       |   |   |   |-- ✅ test_cobyqa.py
|   |   |       |   |   |   |-- ✅ test_constraint_conversion.py
|   |   |       |   |   |   |-- ✅ test_constraints.py
|   |   |       |   |   |   |-- ✅ test_cython_optimize.py
|   |   |       |   |   |   |-- ✅ test_differentiable_functions.py
|   |   |       |   |   |   |-- ✅ test_direct.py
|   |   |       |   |   |   |-- ✅ test_extending.py
|   |   |       |   |   |   |-- ✅ test_hessian_update_strategy.py
|   |   |       |   |   |   |-- ✅ test_isotonic_regression.py
|   |   |       |   |   |   |-- ✅ test_lbfgsb_hessinv.py
|   |   |       |   |   |   |-- ✅ test_lbfgsb_setulb.py
|   |   |       |   |   |   |-- ✅ test_least_squares.py
|   |   |       |   |   |   |-- ✅ test_linear_assignment.py
|   |   |       |   |   |   |-- ✅ test_linesearch.py
|   |   |       |   |   |   |-- ✅ test_linprog.py
|   |   |       |   |   |   |-- ✅ test_lsq_common.py
|   |   |       |   |   |   |-- ✅ test_lsq_linear.py
|   |   |       |   |   |   |-- ✅ test_milp.py
|   |   |       |   |   |   |-- ✅ test_minimize_constrained.py
|   |   |       |   |   |   |-- ✅ test_minpack.py
|   |   |       |   |   |   |-- ✅ test_nnls.py
|   |   |       |   |   |   |-- ✅ test_nonlin.py
|   |   |       |   |   |   |-- ✅ test_optimize.py
|   |   |       |   |   |   |-- ✅ test_quadratic_assignment.py
|   |   |       |   |   |   |-- ✅ test_regression.py
|   |   |       |   |   |   |-- ✅ test_slsqp.py
|   |   |       |   |   |   |-- ✅ test_tnc.py
|   |   |       |   |   |   |-- ✅ test_trustregion.py
|   |   |       |   |   |   |-- ✅ test_trustregion_exact.py
|   |   |       |   |   |   |-- ✅ test_trustregion_krylov.py
|   |   |       |   |   |   \-- ✅ test_zeros.py
|   |   |       |   |   |-- ✅ __init__.pxd
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _basinhopping.py
|   |   |       |   |   |-- ✅ _bglu_dense.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _bglu_dense.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _bracket.py
|   |   |       |   |   |-- ✅ _chandrupatla.py
|   |   |       |   |   |-- ✅ _cobyla_py.py
|   |   |       |   |   |-- ✅ _cobyqa_py.py
|   |   |       |   |   |-- ✅ _constraints.py
|   |   |       |   |   |-- ✅ _dcsrch.py
|   |   |       |   |   |-- ✅ _differentiable_functions.py
|   |   |       |   |   |-- ✅ _differentialevolution.py
|   |   |       |   |   |-- ✅ _direct.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _direct.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _direct_py.py
|   |   |       |   |   |-- ✅ _dual_annealing.py
|   |   |       |   |   |-- ✅ _elementwise.py
|   |   |       |   |   |-- ✅ _group_columns.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _group_columns.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _hessian_update_strategy.py
|   |   |       |   |   |-- ✅ _isotonic.py
|   |   |       |   |   |-- ✅ _lbfgsb.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _lbfgsb.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _lbfgsb_py.py
|   |   |       |   |   |-- ✅ _linesearch.py
|   |   |       |   |   |-- ✅ _linprog.py
|   |   |       |   |   |-- ✅ _linprog_doc.py
|   |   |       |   |   |-- ✅ _linprog_highs.py
|   |   |       |   |   |-- ✅ _linprog_ip.py
|   |   |       |   |   |-- ✅ _linprog_rs.py
|   |   |       |   |   |-- ✅ _linprog_simplex.py
|   |   |       |   |   |-- ✅ _linprog_util.py
|   |   |       |   |   |-- ✅ _lsap.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _lsap.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _milp.py
|   |   |       |   |   |-- ✅ _minimize.py
|   |   |       |   |   |-- ✅ _minpack.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _minpack.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _minpack_py.py
|   |   |       |   |   |-- ✅ _moduleTNC.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _moduleTNC.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _nnls.py
|   |   |       |   |   |-- ✅ _nonlin.py
|   |   |       |   |   |-- ✅ _numdiff.py
|   |   |       |   |   |-- ✅ _optimize.py
|   |   |       |   |   |-- ✅ _pava_pybind.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _pava_pybind.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _qap.py
|   |   |       |   |   |-- ✅ _remove_redundancy.py
|   |   |       |   |   |-- ✅ _root.py
|   |   |       |   |   |-- ✅ _root_scalar.py
|   |   |       |   |   |-- ✅ _shgo.py
|   |   |       |   |   |-- ✅ _slsqp_py.py
|   |   |       |   |   |-- ✅ _slsqplib.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _slsqplib.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _spectral.py
|   |   |       |   |   |-- ✅ _tnc.py
|   |   |       |   |   |-- ✅ _trustregion.py
|   |   |       |   |   |-- ✅ _trustregion_dogleg.py
|   |   |       |   |   |-- ✅ _trustregion_exact.py
|   |   |       |   |   |-- ✅ _trustregion_krylov.py
|   |   |       |   |   |-- ✅ _trustregion_ncg.py
|   |   |       |   |   |-- ✅ _tstutils.py
|   |   |       |   |   |-- ✅ _zeros.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _zeros.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _zeros_py.py
|   |   |       |   |   |-- ✅ cobyla.py
|   |   |       |   |   |-- ✅ cython_optimize.pxd
|   |   |       |   |   |-- ✅ elementwise.py
|   |   |       |   |   |-- ✅ lbfgsb.py
|   |   |       |   |   |-- ✅ linesearch.py
|   |   |       |   |   |-- ✅ minpack.py
|   |   |       |   |   |-- ✅ minpack2.py
|   |   |       |   |   |-- ✅ moduleTNC.py
|   |   |       |   |   |-- ✅ nonlin.py
|   |   |       |   |   |-- ✅ optimize.py
|   |   |       |   |   |-- ✅ slsqp.py
|   |   |       |   |   |-- ✅ tnc.py
|   |   |       |   |   \-- ✅ zeros.py
|   |   |       |   |-- ✅ signal/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _scipy_spectral_test_shim.py
|   |   |       |   |   |   |-- ✅ mpsig.py
|   |   |       |   |   |   |-- ✅ test_array_tools.py
|   |   |       |   |   |   |-- ✅ test_bsplines.py
|   |   |       |   |   |   |-- ✅ test_cont2discrete.py
|   |   |       |   |   |   |-- ✅ test_czt.py
|   |   |       |   |   |   |-- ✅ test_dltisys.py
|   |   |       |   |   |   |-- ✅ test_filter_design.py
|   |   |       |   |   |   |-- ✅ test_fir_filter_design.py
|   |   |       |   |   |   |-- ✅ test_ltisys.py
|   |   |       |   |   |   |-- ✅ test_max_len_seq.py
|   |   |       |   |   |   |-- ✅ test_peak_finding.py
|   |   |       |   |   |   |-- ✅ test_result_type.py
|   |   |       |   |   |   |-- ✅ test_savitzky_golay.py
|   |   |       |   |   |   |-- ✅ test_short_time_fft.py
|   |   |       |   |   |   |-- ✅ test_signaltools.py
|   |   |       |   |   |   |-- ✅ test_spectral.py
|   |   |       |   |   |   |-- ✅ test_splines.py
|   |   |       |   |   |   |-- ✅ test_upfirdn.py
|   |   |       |   |   |   |-- ✅ test_waveforms.py
|   |   |       |   |   |   |-- ✅ test_wavelets.py
|   |   |       |   |   |   \-- ✅ test_windows.py
|   |   |       |   |   |-- ✅ windows/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _windows.py
|   |   |       |   |   |   \-- ✅ windows.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _arraytools.py
|   |   |       |   |   |-- ✅ _czt.py
|   |   |       |   |   |-- ✅ _delegators.py
|   |   |       |   |   |-- ✅ _filter_design.py
|   |   |       |   |   |-- ✅ _fir_filter_design.py
|   |   |       |   |   |-- ✅ _lti_conversion.py
|   |   |       |   |   |-- ✅ _ltisys.py
|   |   |       |   |   |-- ✅ _max_len_seq.py
|   |   |       |   |   |-- ✅ _max_len_seq_inner.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _max_len_seq_inner.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _peak_finding.py
|   |   |       |   |   |-- ✅ _peak_finding_utils.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _peak_finding_utils.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _polyutils.py
|   |   |       |   |   |-- ✅ _savitzky_golay.py
|   |   |       |   |   |-- ✅ _short_time_fft.py
|   |   |       |   |   |-- ✅ _signal_api.py
|   |   |       |   |   |-- ✅ _signaltools.py
|   |   |       |   |   |-- ✅ _sigtools.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _sigtools.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _sosfilt.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _sosfilt.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _spectral_py.py
|   |   |       |   |   |-- ✅ _spline.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _spline.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _spline.pyi
|   |   |       |   |   |-- ✅ _spline_filters.py
|   |   |       |   |   |-- ✅ _support_alternative_backends.py
|   |   |       |   |   |-- ✅ _upfirdn.py
|   |   |       |   |   |-- ✅ _upfirdn_apply.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _upfirdn_apply.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _waveforms.py
|   |   |       |   |   |-- ✅ _wavelets.py
|   |   |       |   |   |-- ✅ bsplines.py
|   |   |       |   |   |-- ✅ filter_design.py
|   |   |       |   |   |-- ✅ fir_filter_design.py
|   |   |       |   |   |-- ✅ lti_conversion.py
|   |   |       |   |   |-- ✅ ltisys.py
|   |   |       |   |   |-- ✅ signaltools.py
|   |   |       |   |   |-- ✅ spectral.py
|   |   |       |   |   |-- ✅ spline.py
|   |   |       |   |   |-- ✅ waveforms.py
|   |   |       |   |   \-- ✅ wavelets.py
|   |   |       |   |-- ✅ sparse/
|   |   |       |   |   |-- ✅ csgraph/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_connected_components.py
|   |   |       |   |   |   |   |-- ✅ test_conversions.py
|   |   |       |   |   |   |   |-- ✅ test_flow.py
|   |   |       |   |   |   |   |-- ✅ test_graph_laplacian.py
|   |   |       |   |   |   |   |-- ✅ test_matching.py
|   |   |       |   |   |   |   |-- ✅ test_pydata_sparse.py
|   |   |       |   |   |   |   |-- ✅ test_reordering.py
|   |   |       |   |   |   |   |-- ✅ test_shortest_path.py
|   |   |       |   |   |   |   |-- ✅ test_spanning_tree.py
|   |   |       |   |   |   |   \-- ✅ test_traversal.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _flow.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _flow.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _laplacian.py
|   |   |       |   |   |   |-- ✅ _matching.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _matching.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _min_spanning_tree.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _min_spanning_tree.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _reordering.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _reordering.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _shortest_path.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _shortest_path.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _tools.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _tools.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _traversal.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _traversal.cp312-win_amd64.pyd
|   |   |       |   |   |   \-- ✅ _validation.py
|   |   |       |   |   |-- ✅ linalg/
|   |   |       |   |   |   |-- ✅ _dsolve/
|   |   |       |   |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   \-- ✅ test_linsolve.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _add_newdocs.py
|   |   |       |   |   |   |   |-- ✅ _superlu.cp312-win_amd64.dll.a
|   |   |       |   |   |   |   |-- ✅ _superlu.cp312-win_amd64.pyd
|   |   |       |   |   |   |   \-- ✅ linsolve.py
|   |   |       |   |   |   |-- ✅ _eigen/
|   |   |       |   |   |   |   |-- ✅ arpack/
|   |   |       |   |   |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   \-- ✅ test_arpack.py
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ _arpack.cp312-win_amd64.dll.a
|   |   |       |   |   |   |   |   |-- ✅ _arpack.cp312-win_amd64.pyd
|   |   |       |   |   |   |   |   |-- ✅ arpack.py
|   |   |       |   |   |   |   |   \-- ✅ COPYING
|   |   |       |   |   |   |   |-- ✅ lobpcg/
|   |   |       |   |   |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   \-- ✅ test_lobpcg.py
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   \-- ✅ lobpcg.py
|   |   |       |   |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   \-- ✅ test_svds.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _svds.py
|   |   |       |   |   |   |   \-- ✅ _svds_doc.py
|   |   |       |   |   |   |-- ✅ _isolve/
|   |   |       |   |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ test_gcrotmk.py
|   |   |       |   |   |   |   |   |-- ✅ test_iterative.py
|   |   |       |   |   |   |   |   |-- ✅ test_lgmres.py
|   |   |       |   |   |   |   |   |-- ✅ test_lsmr.py
|   |   |       |   |   |   |   |   |-- ✅ test_lsqr.py
|   |   |       |   |   |   |   |   |-- ✅ test_minres.py
|   |   |       |   |   |   |   |   \-- ✅ test_utils.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _gcrotmk.py
|   |   |       |   |   |   |   |-- ✅ iterative.py
|   |   |       |   |   |   |   |-- ✅ lgmres.py
|   |   |       |   |   |   |   |-- ✅ lsmr.py
|   |   |       |   |   |   |   |-- ✅ lsqr.py
|   |   |       |   |   |   |   |-- ✅ minres.py
|   |   |       |   |   |   |   |-- ✅ tfqmr.py
|   |   |       |   |   |   |   \-- ✅ utils.py
|   |   |       |   |   |   |-- ✅ _propack/
|   |   |       |   |   |   |   |-- ✅ _cpropack.cp312-win_amd64.dll.a
|   |   |       |   |   |   |   |-- ✅ _cpropack.cp312-win_amd64.pyd
|   |   |       |   |   |   |   |-- ✅ _dpropack.cp312-win_amd64.dll.a
|   |   |       |   |   |   |   |-- ✅ _dpropack.cp312-win_amd64.pyd
|   |   |       |   |   |   |   |-- ✅ _spropack.cp312-win_amd64.dll.a
|   |   |       |   |   |   |   |-- ✅ _spropack.cp312-win_amd64.pyd
|   |   |       |   |   |   |   |-- ✅ _zpropack.cp312-win_amd64.dll.a
|   |   |       |   |   |   |   \-- ✅ _zpropack.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ propack_test_data.npz
|   |   |       |   |   |   |   |-- ✅ test_expm_multiply.py
|   |   |       |   |   |   |   |-- ✅ test_interface.py
|   |   |       |   |   |   |   |-- ✅ test_matfuncs.py
|   |   |       |   |   |   |   |-- ✅ test_norm.py
|   |   |       |   |   |   |   |-- ✅ test_onenormest.py
|   |   |       |   |   |   |   |-- ✅ test_propack.py
|   |   |       |   |   |   |   |-- ✅ test_pydata_sparse.py
|   |   |       |   |   |   |   \-- ✅ test_special_sparse_arrays.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _expm_multiply.py
|   |   |       |   |   |   |-- ✅ _interface.py
|   |   |       |   |   |   |-- ✅ _matfuncs.py
|   |   |       |   |   |   |-- ✅ _norm.py
|   |   |       |   |   |   |-- ✅ _onenormest.py
|   |   |       |   |   |   |-- ✅ _special_sparse_arrays.py
|   |   |       |   |   |   |-- ✅ _svdp.py
|   |   |       |   |   |   |-- ✅ dsolve.py
|   |   |       |   |   |   |-- ✅ eigen.py
|   |   |       |   |   |   |-- ✅ interface.py
|   |   |       |   |   |   |-- ✅ isolve.py
|   |   |       |   |   |   \-- ✅ matfuncs.py
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ csc_py2.npz
|   |   |       |   |   |   |   \-- ✅ csc_py3.npz
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_arithmetic1d.py
|   |   |       |   |   |   |-- ✅ test_array_api.py
|   |   |       |   |   |   |-- ✅ test_base.py
|   |   |       |   |   |   |-- ✅ test_common1d.py
|   |   |       |   |   |   |-- ✅ test_construct.py
|   |   |       |   |   |   |-- ✅ test_coo.py
|   |   |       |   |   |   |-- ✅ test_csc.py
|   |   |       |   |   |   |-- ✅ test_csr.py
|   |   |       |   |   |   |-- ✅ test_dok.py
|   |   |       |   |   |   |-- ✅ test_extract.py
|   |   |       |   |   |   |-- ✅ test_indexing1d.py
|   |   |       |   |   |   |-- ✅ test_matrix_io.py
|   |   |       |   |   |   |-- ✅ test_minmax1d.py
|   |   |       |   |   |   |-- ✅ test_sparsetools.py
|   |   |       |   |   |   |-- ✅ test_spfuncs.py
|   |   |       |   |   |   \-- ✅ test_sputils.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _bsr.py
|   |   |       |   |   |-- ✅ _compressed.py
|   |   |       |   |   |-- ✅ _construct.py
|   |   |       |   |   |-- ✅ _coo.py
|   |   |       |   |   |-- ✅ _csc.py
|   |   |       |   |   |-- ✅ _csparsetools.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _csparsetools.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _csr.py
|   |   |       |   |   |-- ✅ _data.py
|   |   |       |   |   |-- ✅ _dia.py
|   |   |       |   |   |-- ✅ _dok.py
|   |   |       |   |   |-- ✅ _extract.py
|   |   |       |   |   |-- ✅ _index.py
|   |   |       |   |   |-- ✅ _lil.py
|   |   |       |   |   |-- ✅ _matrix.py
|   |   |       |   |   |-- ✅ _matrix_io.py
|   |   |       |   |   |-- ✅ _sparsetools.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _sparsetools.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _spfuncs.py
|   |   |       |   |   |-- ✅ _sputils.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ bsr.py
|   |   |       |   |   |-- ✅ compressed.py
|   |   |       |   |   |-- ✅ construct.py
|   |   |       |   |   |-- ✅ coo.py
|   |   |       |   |   |-- ✅ csc.py
|   |   |       |   |   |-- ✅ csr.py
|   |   |       |   |   |-- ✅ data.py
|   |   |       |   |   |-- ✅ dia.py
|   |   |       |   |   |-- ✅ dok.py
|   |   |       |   |   |-- ✅ extract.py
|   |   |       |   |   |-- ✅ lil.py
|   |   |       |   |   |-- ✅ sparsetools.py
|   |   |       |   |   |-- ✅ spfuncs.py
|   |   |       |   |   \-- ✅ sputils.py
|   |   |       |   |-- ✅ spatial/
|   |   |       |   |   |-- ✅ qhull_src/
|   |   |       |   |   |   \-- ✅ COPYING_QHULL.txt
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ cdist-X1.txt
|   |   |       |   |   |   |   |-- ✅ cdist-X2.txt
|   |   |       |   |   |   |   |-- ✅ degenerate_pointset.npz
|   |   |       |   |   |   |   |-- ✅ iris.txt
|   |   |       |   |   |   |   |-- ✅ pdist-boolean-inp.txt
|   |   |       |   |   |   |   |-- ✅ pdist-chebyshev-ml-iris.txt
|   |   |       |   |   |   |   |-- ✅ pdist-chebyshev-ml.txt
|   |   |       |   |   |   |   |-- ✅ pdist-cityblock-ml-iris.txt
|   |   |       |   |   |   |   |-- ✅ pdist-cityblock-ml.txt
|   |   |       |   |   |   |   |-- ✅ pdist-correlation-ml-iris.txt
|   |   |       |   |   |   |   |-- ✅ pdist-correlation-ml.txt
|   |   |       |   |   |   |   |-- ✅ pdist-cosine-ml-iris.txt
|   |   |       |   |   |   |   |-- ✅ pdist-cosine-ml.txt
|   |   |       |   |   |   |   |-- ✅ pdist-double-inp.txt
|   |   |       |   |   |   |   |-- ✅ pdist-euclidean-ml-iris.txt
|   |   |       |   |   |   |   |-- ✅ pdist-euclidean-ml.txt
|   |   |       |   |   |   |   |-- ✅ pdist-hamming-ml.txt
|   |   |       |   |   |   |   |-- ✅ pdist-jaccard-ml.txt
|   |   |       |   |   |   |   |-- ✅ pdist-jensenshannon-ml-iris.txt
|   |   |       |   |   |   |   |-- ✅ pdist-jensenshannon-ml.txt
|   |   |       |   |   |   |   |-- ✅ pdist-minkowski-3.2-ml-iris.txt
|   |   |       |   |   |   |   |-- ✅ pdist-minkowski-3.2-ml.txt
|   |   |       |   |   |   |   |-- ✅ pdist-minkowski-5.8-ml-iris.txt
|   |   |       |   |   |   |   |-- ✅ pdist-seuclidean-ml-iris.txt
|   |   |       |   |   |   |   |-- ✅ pdist-seuclidean-ml.txt
|   |   |       |   |   |   |   |-- ✅ pdist-spearman-ml.txt
|   |   |       |   |   |   |   |-- ✅ random-bool-data.txt
|   |   |       |   |   |   |   |-- ✅ random-double-data.txt
|   |   |       |   |   |   |   |-- ✅ random-int-data.txt
|   |   |       |   |   |   |   |-- ✅ random-uint-data.txt
|   |   |       |   |   |   |   \-- ✅ selfdual-4d-polytope.txt
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test__plotutils.py
|   |   |       |   |   |   |-- ✅ test__procrustes.py
|   |   |       |   |   |   |-- ✅ test_distance.py
|   |   |       |   |   |   |-- ✅ test_hausdorff.py
|   |   |       |   |   |   |-- ✅ test_kdtree.py
|   |   |       |   |   |   |-- ✅ test_qhull.py
|   |   |       |   |   |   |-- ✅ test_slerp.py
|   |   |       |   |   |   \-- ✅ test_spherical_voronoi.py
|   |   |       |   |   |-- ✅ transform/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_rigid_transform.py
|   |   |       |   |   |   |   |-- ✅ test_rotation.py
|   |   |       |   |   |   |   |-- ✅ test_rotation_groups.py
|   |   |       |   |   |   |   \-- ✅ test_rotation_spline.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _rigid_transform.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _rigid_transform.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _rotation.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ _rotation.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _rotation_groups.py
|   |   |       |   |   |   |-- ✅ _rotation_spline.py
|   |   |       |   |   |   \-- ✅ rotation.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _ckdtree.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _ckdtree.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _distance_pybind.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _distance_pybind.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _distance_wrap.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _distance_wrap.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _geometric_slerp.py
|   |   |       |   |   |-- ✅ _hausdorff.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _hausdorff.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _kdtree.py
|   |   |       |   |   |-- ✅ _plotutils.py
|   |   |       |   |   |-- ✅ _procrustes.py
|   |   |       |   |   |-- ✅ _qhull.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _qhull.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _qhull.pyi
|   |   |       |   |   |-- ✅ _spherical_voronoi.py
|   |   |       |   |   |-- ✅ _voronoi.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _voronoi.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _voronoi.pyi
|   |   |       |   |   |-- ✅ ckdtree.py
|   |   |       |   |   |-- ✅ distance.py
|   |   |       |   |   |-- ✅ distance.pyi
|   |   |       |   |   |-- ✅ kdtree.py
|   |   |       |   |   \-- ✅ qhull.py
|   |   |       |   |-- ✅ special/
|   |   |       |   |   |-- ✅ _precompute/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ cosine_cdf.py
|   |   |       |   |   |   |-- ✅ expn_asy.py
|   |   |       |   |   |   |-- ✅ gammainc_asy.py
|   |   |       |   |   |   |-- ✅ gammainc_data.py
|   |   |       |   |   |   |-- ✅ hyp2f1_data.py
|   |   |       |   |   |   |-- ✅ lambertw.py
|   |   |       |   |   |   |-- ✅ loggamma.py
|   |   |       |   |   |   |-- ✅ struve_convergence.py
|   |   |       |   |   |   |-- ✅ utils.py
|   |   |       |   |   |   |-- ✅ wright_bessel.py
|   |   |       |   |   |   |-- ✅ wright_bessel_data.py
|   |   |       |   |   |   |-- ✅ wrightomega.py
|   |   |       |   |   |   \-- ✅ zetac.py
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ _cython_examples/
|   |   |       |   |   |   |   |-- ✅ extending.pyx
|   |   |       |   |   |   |   \-- ✅ meson.build
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ boost.npz
|   |   |       |   |   |   |   |-- ✅ gsl.npz
|   |   |       |   |   |   |   \-- ✅ local.npz
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_basic.py
|   |   |       |   |   |   |-- ✅ test_bdtr.py
|   |   |       |   |   |   |-- ✅ test_boost_ufuncs.py
|   |   |       |   |   |   |-- ✅ test_boxcox.py
|   |   |       |   |   |   |-- ✅ test_cdflib.py
|   |   |       |   |   |   |-- ✅ test_cdft_asymptotic.py
|   |   |       |   |   |   |-- ✅ test_cephes_intp_cast.py
|   |   |       |   |   |   |-- ✅ test_cosine_distr.py
|   |   |       |   |   |   |-- ✅ test_cython_special.py
|   |   |       |   |   |   |-- ✅ test_data.py
|   |   |       |   |   |   |-- ✅ test_dd.py
|   |   |       |   |   |   |-- ✅ test_digamma.py
|   |   |       |   |   |   |-- ✅ test_ellip_harm.py
|   |   |       |   |   |   |-- ✅ test_erfinv.py
|   |   |       |   |   |   |-- ✅ test_exponential_integrals.py
|   |   |       |   |   |   |-- ✅ test_extending.py
|   |   |       |   |   |   |-- ✅ test_faddeeva.py
|   |   |       |   |   |   |-- ✅ test_gamma.py
|   |   |       |   |   |   |-- ✅ test_gammainc.py
|   |   |       |   |   |   |-- ✅ test_hyp2f1.py
|   |   |       |   |   |   |-- ✅ test_hypergeometric.py
|   |   |       |   |   |   |-- ✅ test_iv_ratio.py
|   |   |       |   |   |   |-- ✅ test_kolmogorov.py
|   |   |       |   |   |   |-- ✅ test_lambertw.py
|   |   |       |   |   |   |-- ✅ test_legendre.py
|   |   |       |   |   |   |-- ✅ test_log1mexp.py
|   |   |       |   |   |   |-- ✅ test_loggamma.py
|   |   |       |   |   |   |-- ✅ test_logit.py
|   |   |       |   |   |   |-- ✅ test_logsumexp.py
|   |   |       |   |   |   |-- ✅ test_mpmath.py
|   |   |       |   |   |   |-- ✅ test_nan_inputs.py
|   |   |       |   |   |   |-- ✅ test_ndtr.py
|   |   |       |   |   |   |-- ✅ test_ndtri_exp.py
|   |   |       |   |   |   |-- ✅ test_orthogonal.py
|   |   |       |   |   |   |-- ✅ test_orthogonal_eval.py
|   |   |       |   |   |   |-- ✅ test_owens_t.py
|   |   |       |   |   |   |-- ✅ test_pcf.py
|   |   |       |   |   |   |-- ✅ test_pdtr.py
|   |   |       |   |   |   |-- ✅ test_powm1.py
|   |   |       |   |   |   |-- ✅ test_precompute_expn_asy.py
|   |   |       |   |   |   |-- ✅ test_precompute_gammainc.py
|   |   |       |   |   |   |-- ✅ test_precompute_utils.py
|   |   |       |   |   |   |-- ✅ test_round.py
|   |   |       |   |   |   |-- ✅ test_sf_error.py
|   |   |       |   |   |   |-- ✅ test_sici.py
|   |   |       |   |   |   |-- ✅ test_specfun.py
|   |   |       |   |   |   |-- ✅ test_spence.py
|   |   |       |   |   |   |-- ✅ test_spfun_stats.py
|   |   |       |   |   |   |-- ✅ test_sph_harm.py
|   |   |       |   |   |   |-- ✅ test_spherical_bessel.py
|   |   |       |   |   |   |-- ✅ test_support_alternative_backends.py
|   |   |       |   |   |   |-- ✅ test_trig.py
|   |   |       |   |   |   |-- ✅ test_ufunc_signatures.py
|   |   |       |   |   |   |-- ✅ test_wright_bessel.py
|   |   |       |   |   |   |-- ✅ test_wrightomega.py
|   |   |       |   |   |   \-- ✅ test_zeta.py
|   |   |       |   |   |-- ✅ __init__.pxd
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _add_newdocs.py
|   |   |       |   |   |-- ✅ _basic.py
|   |   |       |   |   |-- ✅ _comb.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _comb.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _ellip_harm.py
|   |   |       |   |   |-- ✅ _ellip_harm_2.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _ellip_harm_2.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _gufuncs.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _gufuncs.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _input_validation.py
|   |   |       |   |   |-- ✅ _lambertw.py
|   |   |       |   |   |-- ✅ _logsumexp.py
|   |   |       |   |   |-- ✅ _mptestutils.py
|   |   |       |   |   |-- ✅ _multiufuncs.py
|   |   |       |   |   |-- ✅ _orthogonal.py
|   |   |       |   |   |-- ✅ _orthogonal.pyi
|   |   |       |   |   |-- ✅ _sf_error.py
|   |   |       |   |   |-- ✅ _specfun.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _specfun.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _special_ufuncs.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _special_ufuncs.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _spfun_stats.py
|   |   |       |   |   |-- ✅ _spherical_bessel.py
|   |   |       |   |   |-- ✅ _support_alternative_backends.py
|   |   |       |   |   |-- ✅ _test_internal.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _test_internal.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _test_internal.pyi
|   |   |       |   |   |-- ✅ _testutils.py
|   |   |       |   |   |-- ✅ _ufuncs.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _ufuncs.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _ufuncs.pyi
|   |   |       |   |   |-- ✅ _ufuncs.pyx
|   |   |       |   |   |-- ✅ _ufuncs_cxx.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _ufuncs_cxx.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _ufuncs_cxx.pxd
|   |   |       |   |   |-- ✅ _ufuncs_cxx.pyx
|   |   |       |   |   |-- ✅ _ufuncs_cxx_defs.h
|   |   |       |   |   |-- ✅ _ufuncs_defs.h
|   |   |       |   |   |-- ✅ add_newdocs.py
|   |   |       |   |   |-- ✅ basic.py
|   |   |       |   |   |-- ✅ cython_special.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ cython_special.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ cython_special.pxd
|   |   |       |   |   |-- ✅ cython_special.pyi
|   |   |       |   |   |-- ✅ orthogonal.py
|   |   |       |   |   |-- ✅ sf_error.py
|   |   |       |   |   |-- ✅ specfun.py
|   |   |       |   |   \-- ✅ spfun_stats.py
|   |   |       |   |-- ✅ stats/
|   |   |       |   |   |-- ✅ _levy_stable/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ levyst.cp312-win_amd64.dll.a
|   |   |       |   |   |   \-- ✅ levyst.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _rcont/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ rcont.cp312-win_amd64.dll.a
|   |   |       |   |   |   \-- ✅ rcont.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _unuran/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ unuran_wrapper.cp312-win_amd64.dll.a
|   |   |       |   |   |   |-- ✅ unuran_wrapper.cp312-win_amd64.pyd
|   |   |       |   |   |   \-- ✅ unuran_wrapper.pyi
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ levy_stable/
|   |   |       |   |   |   |   |   |-- ✅ stable-loc-scale-sample-data.npy
|   |   |       |   |   |   |   |   |-- ✅ stable-Z1-cdf-sample-data.npy
|   |   |       |   |   |   |   |   \-- ✅ stable-Z1-pdf-sample-data.npy
|   |   |       |   |   |   |   |-- ✅ nist_anova/
|   |   |       |   |   |   |   |   |-- ✅ AtmWtAg.dat
|   |   |       |   |   |   |   |   |-- ✅ SiRstv.dat
|   |   |       |   |   |   |   |   |-- ✅ SmLs01.dat
|   |   |       |   |   |   |   |   |-- ✅ SmLs02.dat
|   |   |       |   |   |   |   |   |-- ✅ SmLs03.dat
|   |   |       |   |   |   |   |   |-- ✅ SmLs04.dat
|   |   |       |   |   |   |   |   |-- ✅ SmLs05.dat
|   |   |       |   |   |   |   |   |-- ✅ SmLs06.dat
|   |   |       |   |   |   |   |   |-- ✅ SmLs07.dat
|   |   |       |   |   |   |   |   |-- ✅ SmLs08.dat
|   |   |       |   |   |   |   |   \-- ✅ SmLs09.dat
|   |   |       |   |   |   |   |-- ✅ nist_linregress/
|   |   |       |   |   |   |   |   \-- ✅ Norris.dat
|   |   |       |   |   |   |   |-- ✅ _mvt.py
|   |   |       |   |   |   |   |-- ✅ fisher_exact_results_from_r.py
|   |   |       |   |   |   |   |-- ✅ jf_skew_t_gamlss_pdf_data.npy
|   |   |       |   |   |   |   |-- ✅ rel_breitwigner_pdf_sample_data_ROOT.npy
|   |   |       |   |   |   |   \-- ✅ studentized_range_mpmath_ref.json
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common_tests.py
|   |   |       |   |   |   |-- ✅ test_axis_nan_policy.py
|   |   |       |   |   |   |-- ✅ test_binned_statistic.py
|   |   |       |   |   |   |-- ✅ test_censored_data.py
|   |   |       |   |   |   |-- ✅ test_contingency.py
|   |   |       |   |   |   |-- ✅ test_continued_fraction.py
|   |   |       |   |   |   |-- ✅ test_continuous.py
|   |   |       |   |   |   |-- ✅ test_continuous_basic.py
|   |   |       |   |   |   |-- ✅ test_continuous_fit_censored.py
|   |   |       |   |   |   |-- ✅ test_correlation.py
|   |   |       |   |   |   |-- ✅ test_crosstab.py
|   |   |       |   |   |   |-- ✅ test_discrete_basic.py
|   |   |       |   |   |   |-- ✅ test_discrete_distns.py
|   |   |       |   |   |   |-- ✅ test_distributions.py
|   |   |       |   |   |   |-- ✅ test_entropy.py
|   |   |       |   |   |   |-- ✅ test_fast_gen_inversion.py
|   |   |       |   |   |   |-- ✅ test_fit.py
|   |   |       |   |   |   |-- ✅ test_hypotests.py
|   |   |       |   |   |   |-- ✅ test_kdeoth.py
|   |   |       |   |   |   |-- ✅ test_marray.py
|   |   |       |   |   |   |-- ✅ test_mgc.py
|   |   |       |   |   |   |-- ✅ test_morestats.py
|   |   |       |   |   |   |-- ✅ test_mstats_basic.py
|   |   |       |   |   |   |-- ✅ test_mstats_extras.py
|   |   |       |   |   |   |-- ✅ test_multicomp.py
|   |   |       |   |   |   |-- ✅ test_multivariate.py
|   |   |       |   |   |   |-- ✅ test_odds_ratio.py
|   |   |       |   |   |   |-- ✅ test_qmc.py
|   |   |       |   |   |   |-- ✅ test_quantile.py
|   |   |       |   |   |   |-- ✅ test_rank.py
|   |   |       |   |   |   |-- ✅ test_relative_risk.py
|   |   |       |   |   |   |-- ✅ test_resampling.py
|   |   |       |   |   |   |-- ✅ test_sampling.py
|   |   |       |   |   |   |-- ✅ test_sensitivity_analysis.py
|   |   |       |   |   |   |-- ✅ test_stats.py
|   |   |       |   |   |   |-- ✅ test_survival.py
|   |   |       |   |   |   |-- ✅ test_tukeylambda_stats.py
|   |   |       |   |   |   \-- ✅ test_variation.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _ansari_swilk_statistics.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _ansari_swilk_statistics.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _axis_nan_policy.py
|   |   |       |   |   |-- ✅ _biasedurn.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _biasedurn.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _biasedurn.pxd
|   |   |       |   |   |-- ✅ _binned_statistic.py
|   |   |       |   |   |-- ✅ _binomtest.py
|   |   |       |   |   |-- ✅ _bws_test.py
|   |   |       |   |   |-- ✅ _censored_data.py
|   |   |       |   |   |-- ✅ _common.py
|   |   |       |   |   |-- ✅ _constants.py
|   |   |       |   |   |-- ✅ _continued_fraction.py
|   |   |       |   |   |-- ✅ _continuous_distns.py
|   |   |       |   |   |-- ✅ _correlation.py
|   |   |       |   |   |-- ✅ _covariance.py
|   |   |       |   |   |-- ✅ _crosstab.py
|   |   |       |   |   |-- ✅ _discrete_distns.py
|   |   |       |   |   |-- ✅ _distn_infrastructure.py
|   |   |       |   |   |-- ✅ _distr_params.py
|   |   |       |   |   |-- ✅ _distribution_infrastructure.py
|   |   |       |   |   |-- ✅ _entropy.py
|   |   |       |   |   |-- ✅ _finite_differences.py
|   |   |       |   |   |-- ✅ _fit.py
|   |   |       |   |   |-- ✅ _hypotests.py
|   |   |       |   |   |-- ✅ _kde.py
|   |   |       |   |   |-- ✅ _ksstats.py
|   |   |       |   |   |-- ✅ _mannwhitneyu.py
|   |   |       |   |   |-- ✅ _mgc.py
|   |   |       |   |   |-- ✅ _morestats.py
|   |   |       |   |   |-- ✅ _mstats_basic.py
|   |   |       |   |   |-- ✅ _mstats_extras.py
|   |   |       |   |   |-- ✅ _multicomp.py
|   |   |       |   |   |-- ✅ _multivariate.py
|   |   |       |   |   |-- ✅ _new_distributions.py
|   |   |       |   |   |-- ✅ _odds_ratio.py
|   |   |       |   |   |-- ✅ _page_trend_test.py
|   |   |       |   |   |-- ✅ _probability_distribution.py
|   |   |       |   |   |-- ✅ _qmc.py
|   |   |       |   |   |-- ✅ _qmc_cy.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _qmc_cy.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _qmc_cy.pyi
|   |   |       |   |   |-- ✅ _qmvnt.py
|   |   |       |   |   |-- ✅ _qmvnt_cy.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _qmvnt_cy.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _quantile.py
|   |   |       |   |   |-- ✅ _relative_risk.py
|   |   |       |   |   |-- ✅ _resampling.py
|   |   |       |   |   |-- ✅ _result_classes.py
|   |   |       |   |   |-- ✅ _sampling.py
|   |   |       |   |   |-- ✅ _sensitivity_analysis.py
|   |   |       |   |   |-- ✅ _sobol.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _sobol.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _sobol.pyi
|   |   |       |   |   |-- ✅ _sobol_direction_numbers.npz
|   |   |       |   |   |-- ✅ _stats.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _stats.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _stats.pxd
|   |   |       |   |   |-- ✅ _stats_mstats_common.py
|   |   |       |   |   |-- ✅ _stats_py.py
|   |   |       |   |   |-- ✅ _stats_pythran.cp312-win_amd64.dll.a
|   |   |       |   |   |-- ✅ _stats_pythran.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _survival.py
|   |   |       |   |   |-- ✅ _tukeylambda_stats.py
|   |   |       |   |   |-- ✅ _variation.py
|   |   |       |   |   |-- ✅ _warnings_errors.py
|   |   |       |   |   |-- ✅ _wilcoxon.py
|   |   |       |   |   |-- ✅ biasedurn.py
|   |   |       |   |   |-- ✅ contingency.py
|   |   |       |   |   |-- ✅ distributions.py
|   |   |       |   |   |-- ✅ kde.py
|   |   |       |   |   |-- ✅ morestats.py
|   |   |       |   |   |-- ✅ mstats.py
|   |   |       |   |   |-- ✅ mstats_basic.py
|   |   |       |   |   |-- ✅ mstats_extras.py
|   |   |       |   |   |-- ✅ mvn.py
|   |   |       |   |   |-- ✅ qmc.py
|   |   |       |   |   |-- ✅ sampling.py
|   |   |       |   |   \-- ✅ stats.py
|   |   |       |   |-- ✅ __config__.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _cyutility.cp312-win_amd64.dll.a
|   |   |       |   |-- ✅ _cyutility.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _distributor_init.py
|   |   |       |   |-- ✅ conftest.py
|   |   |       |   \-- ✅ version.py
|   |   |       |-- ✅ scipy-1.16.2.dist-info/
|   |   |       |   |-- ✅ DELVEWHEEL
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ scipy.libs/
|   |   |       |   \-- ✅ libscipy_openblas-48c358d105077551cc9cc3ba79387ed5.dll
|   |   |       |-- ✅ six-1.17.0.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ sklearn/
|   |   |       |   |-- ✅ .libs/
|   |   |       |   |   |-- ✅ msvcp140.dll
|   |   |       |   |   \-- ✅ vcomp140.dll
|   |   |       |   |-- ✅ __check_build/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _check_build.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _check_build.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _check_build.pyx
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ _build_utils/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ tempita.py
|   |   |       |   |   \-- ✅ version.py
|   |   |       |   |-- ✅ _loss/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_link.py
|   |   |       |   |   |   \-- ✅ test_loss.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _loss.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _loss.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _loss.pxd
|   |   |       |   |   |-- ✅ _loss.pyx.tp
|   |   |       |   |   |-- ✅ link.py
|   |   |       |   |   |-- ✅ loss.py
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ cluster/
|   |   |       |   |   |-- ✅ _hdbscan/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ test_reachibility.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _linkage.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _linkage.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _linkage.pyx
|   |   |       |   |   |   |-- ✅ _reachability.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _reachability.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _reachability.pyx
|   |   |       |   |   |   |-- ✅ _tree.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _tree.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _tree.pxd
|   |   |       |   |   |   |-- ✅ _tree.pyx
|   |   |       |   |   |   |-- ✅ hdbscan.py
|   |   |       |   |   |   \-- ✅ meson.build
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ test_affinity_propagation.py
|   |   |       |   |   |   |-- ✅ test_bicluster.py
|   |   |       |   |   |   |-- ✅ test_birch.py
|   |   |       |   |   |   |-- ✅ test_bisect_k_means.py
|   |   |       |   |   |   |-- ✅ test_dbscan.py
|   |   |       |   |   |   |-- ✅ test_feature_agglomeration.py
|   |   |       |   |   |   |-- ✅ test_hdbscan.py
|   |   |       |   |   |   |-- ✅ test_hierarchical.py
|   |   |       |   |   |   |-- ✅ test_k_means.py
|   |   |       |   |   |   |-- ✅ test_mean_shift.py
|   |   |       |   |   |   |-- ✅ test_optics.py
|   |   |       |   |   |   \-- ✅ test_spectral.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _affinity_propagation.py
|   |   |       |   |   |-- ✅ _agglomerative.py
|   |   |       |   |   |-- ✅ _bicluster.py
|   |   |       |   |   |-- ✅ _birch.py
|   |   |       |   |   |-- ✅ _bisect_k_means.py
|   |   |       |   |   |-- ✅ _dbscan.py
|   |   |       |   |   |-- ✅ _dbscan_inner.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _dbscan_inner.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _dbscan_inner.pyx
|   |   |       |   |   |-- ✅ _feature_agglomeration.py
|   |   |       |   |   |-- ✅ _hierarchical_fast.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _hierarchical_fast.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _hierarchical_fast.pxd
|   |   |       |   |   |-- ✅ _hierarchical_fast.pyx
|   |   |       |   |   |-- ✅ _k_means_common.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _k_means_common.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _k_means_common.pxd
|   |   |       |   |   |-- ✅ _k_means_common.pyx
|   |   |       |   |   |-- ✅ _k_means_elkan.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _k_means_elkan.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _k_means_elkan.pyx
|   |   |       |   |   |-- ✅ _k_means_lloyd.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _k_means_lloyd.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _k_means_lloyd.pyx
|   |   |       |   |   |-- ✅ _k_means_minibatch.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _k_means_minibatch.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _k_means_minibatch.pyx
|   |   |       |   |   |-- ✅ _kmeans.py
|   |   |       |   |   |-- ✅ _mean_shift.py
|   |   |       |   |   |-- ✅ _optics.py
|   |   |       |   |   |-- ✅ _spectral.py
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ compose/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_column_transformer.py
|   |   |       |   |   |   \-- ✅ test_target.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _column_transformer.py
|   |   |       |   |   \-- ✅ _target.py
|   |   |       |   |-- ✅ covariance/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_covariance.py
|   |   |       |   |   |   |-- ✅ test_elliptic_envelope.py
|   |   |       |   |   |   |-- ✅ test_graphical_lasso.py
|   |   |       |   |   |   \-- ✅ test_robust_covariance.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _elliptic_envelope.py
|   |   |       |   |   |-- ✅ _empirical_covariance.py
|   |   |       |   |   |-- ✅ _graph_lasso.py
|   |   |       |   |   |-- ✅ _robust_covariance.py
|   |   |       |   |   \-- ✅ _shrunk_covariance.py
|   |   |       |   |-- ✅ cross_decomposition/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ test_pls.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ _pls.py
|   |   |       |   |-- ✅ datasets/
|   |   |       |   |   |-- ✅ data/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ breast_cancer.csv
|   |   |       |   |   |   |-- ✅ diabetes_data_raw.csv.gz
|   |   |       |   |   |   |-- ✅ diabetes_target.csv.gz
|   |   |       |   |   |   |-- ✅ digits.csv.gz
|   |   |       |   |   |   |-- ✅ iris.csv
|   |   |       |   |   |   |-- ✅ linnerud_exercise.csv
|   |   |       |   |   |   |-- ✅ linnerud_physiological.csv
|   |   |       |   |   |   \-- ✅ wine_data.csv
|   |   |       |   |   |-- ✅ descr/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ breast_cancer.rst
|   |   |       |   |   |   |-- ✅ california_housing.rst
|   |   |       |   |   |   |-- ✅ covtype.rst
|   |   |       |   |   |   |-- ✅ diabetes.rst
|   |   |       |   |   |   |-- ✅ digits.rst
|   |   |       |   |   |   |-- ✅ iris.rst
|   |   |       |   |   |   |-- ✅ kddcup99.rst
|   |   |       |   |   |   |-- ✅ lfw.rst
|   |   |       |   |   |   |-- ✅ linnerud.rst
|   |   |       |   |   |   |-- ✅ olivetti_faces.rst
|   |   |       |   |   |   |-- ✅ rcv1.rst
|   |   |       |   |   |   |-- ✅ species_distributions.rst
|   |   |       |   |   |   |-- ✅ twenty_newsgroups.rst
|   |   |       |   |   |   \-- ✅ wine_data.rst
|   |   |       |   |   |-- ✅ images/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ china.jpg
|   |   |       |   |   |   |-- ✅ flower.jpg
|   |   |       |   |   |   \-- ✅ README.txt
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ data/
|   |   |       |   |   |   |   |-- ✅ openml/
|   |   |       |   |   |   |   |   |-- ✅ id_1/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-1.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-1.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-1.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-1.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_1119/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-1119.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-1119.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-adult-census-l-2-dv-1.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-adult-census-l-2-s-act-.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-1119.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-54002.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_1590/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-1590.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-1590.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-1590.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-1595261.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_2/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-2.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-2.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-anneal-l-2-dv-1.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-anneal-l-2-s-act-.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-2.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-1666876.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_292/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-292.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-40981.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-292.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-40981.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-australian-l-2-dv-1-s-dact.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-australian-l-2-dv-1.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-australian-l-2-s-act-.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-49822.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_3/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-3.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-3.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-3.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-3.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_40589/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-40589.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-40589.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-emotions-l-2-dv-3.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-emotions-l-2-s-act-.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-40589.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-4644182.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_40675/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-40675.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-40675.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-glass2-l-2-dv-1-s-dact.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-glass2-l-2-dv-1.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-glass2-l-2-s-act-.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-40675.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-4965250.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_40945/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-40945.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-40945.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-40945.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-16826755.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_40966/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-40966.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-40966.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-miceprotein-l-2-dv-4.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-miceprotein-l-2-s-act-.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-40966.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-17928620.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_42074/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-42074.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-42074.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-42074.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-21552912.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_42585/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-42585.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-42585.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-42585.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-21854866.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_561/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-561.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-561.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-cpu-l-2-dv-1.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-cpu-l-2-s-act-.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-561.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-52739.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_61/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-61.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-61.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-iris-l-2-dv-1.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdl-dn-iris-l-2-s-act-.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-61.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-61.arff.gz
|   |   |       |   |   |   |   |   |-- ✅ id_62/
|   |   |       |   |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jd-62.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdf-62.json.gz
|   |   |       |   |   |   |   |   |   |-- ✅ api-v1-jdq-62.json.gz
|   |   |       |   |   |   |   |   |   \-- ✅ data-v1-dl-52352.arff.gz
|   |   |       |   |   |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ svmlight_classification.txt
|   |   |       |   |   |   |   |-- ✅ svmlight_invalid.txt
|   |   |       |   |   |   |   |-- ✅ svmlight_invalid_order.txt
|   |   |       |   |   |   |   \-- ✅ svmlight_multilabel.txt
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_20news.py
|   |   |       |   |   |   |-- ✅ test_arff_parser.py
|   |   |       |   |   |   |-- ✅ test_base.py
|   |   |       |   |   |   |-- ✅ test_california_housing.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_covtype.py
|   |   |       |   |   |   |-- ✅ test_kddcup99.py
|   |   |       |   |   |   |-- ✅ test_lfw.py
|   |   |       |   |   |   |-- ✅ test_olivetti_faces.py
|   |   |       |   |   |   |-- ✅ test_openml.py
|   |   |       |   |   |   |-- ✅ test_rcv1.py
|   |   |       |   |   |   |-- ✅ test_samples_generator.py
|   |   |       |   |   |   \-- ✅ test_svmlight_format.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _arff_parser.py
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _california_housing.py
|   |   |       |   |   |-- ✅ _covtype.py
|   |   |       |   |   |-- ✅ _kddcup99.py
|   |   |       |   |   |-- ✅ _lfw.py
|   |   |       |   |   |-- ✅ _olivetti_faces.py
|   |   |       |   |   |-- ✅ _openml.py
|   |   |       |   |   |-- ✅ _rcv1.py
|   |   |       |   |   |-- ✅ _samples_generator.py
|   |   |       |   |   |-- ✅ _species_distributions.py
|   |   |       |   |   |-- ✅ _svmlight_format_fast.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _svmlight_format_fast.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _svmlight_format_fast.pyx
|   |   |       |   |   |-- ✅ _svmlight_format_io.py
|   |   |       |   |   |-- ✅ _twenty_newsgroups.py
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ decomposition/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_dict_learning.py
|   |   |       |   |   |   |-- ✅ test_factor_analysis.py
|   |   |       |   |   |   |-- ✅ test_fastica.py
|   |   |       |   |   |   |-- ✅ test_incremental_pca.py
|   |   |       |   |   |   |-- ✅ test_kernel_pca.py
|   |   |       |   |   |   |-- ✅ test_nmf.py
|   |   |       |   |   |   |-- ✅ test_online_lda.py
|   |   |       |   |   |   |-- ✅ test_pca.py
|   |   |       |   |   |   |-- ✅ test_sparse_pca.py
|   |   |       |   |   |   \-- ✅ test_truncated_svd.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _cdnmf_fast.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _cdnmf_fast.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _cdnmf_fast.pyx
|   |   |       |   |   |-- ✅ _dict_learning.py
|   |   |       |   |   |-- ✅ _factor_analysis.py
|   |   |       |   |   |-- ✅ _fastica.py
|   |   |       |   |   |-- ✅ _incremental_pca.py
|   |   |       |   |   |-- ✅ _kernel_pca.py
|   |   |       |   |   |-- ✅ _lda.py
|   |   |       |   |   |-- ✅ _nmf.py
|   |   |       |   |   |-- ✅ _online_lda_fast.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _online_lda_fast.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _online_lda_fast.pyx
|   |   |       |   |   |-- ✅ _pca.py
|   |   |       |   |   |-- ✅ _sparse_pca.py
|   |   |       |   |   |-- ✅ _truncated_svd.py
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ ensemble/
|   |   |       |   |   |-- ✅ _hist_gradient_boosting/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_binning.py
|   |   |       |   |   |   |   |-- ✅ test_bitset.py
|   |   |       |   |   |   |   |-- ✅ test_compare_lightgbm.py
|   |   |       |   |   |   |   |-- ✅ test_gradient_boosting.py
|   |   |       |   |   |   |   |-- ✅ test_grower.py
|   |   |       |   |   |   |   |-- ✅ test_histogram.py
|   |   |       |   |   |   |   |-- ✅ test_monotonic_constraints.py
|   |   |       |   |   |   |   |-- ✅ test_predictor.py
|   |   |       |   |   |   |   |-- ✅ test_splitting.py
|   |   |       |   |   |   |   \-- ✅ test_warm_start.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _binning.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _binning.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _binning.pyx
|   |   |       |   |   |   |-- ✅ _bitset.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _bitset.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _bitset.pxd
|   |   |       |   |   |   |-- ✅ _bitset.pyx
|   |   |       |   |   |   |-- ✅ _gradient_boosting.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _gradient_boosting.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _gradient_boosting.pyx
|   |   |       |   |   |   |-- ✅ _predictor.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _predictor.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _predictor.pyx
|   |   |       |   |   |   |-- ✅ binning.py
|   |   |       |   |   |   |-- ✅ common.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ common.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ common.pxd
|   |   |       |   |   |   |-- ✅ common.pyx
|   |   |       |   |   |   |-- ✅ gradient_boosting.py
|   |   |       |   |   |   |-- ✅ grower.py
|   |   |       |   |   |   |-- ✅ histogram.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ histogram.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ histogram.pyx
|   |   |       |   |   |   |-- ✅ meson.build
|   |   |       |   |   |   |-- ✅ predictor.py
|   |   |       |   |   |   |-- ✅ splitting.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ splitting.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ splitting.pyx
|   |   |       |   |   |   \-- ✅ utils.py
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_bagging.py
|   |   |       |   |   |   |-- ✅ test_base.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_forest.py
|   |   |       |   |   |   |-- ✅ test_gradient_boosting.py
|   |   |       |   |   |   |-- ✅ test_iforest.py
|   |   |       |   |   |   |-- ✅ test_stacking.py
|   |   |       |   |   |   |-- ✅ test_voting.py
|   |   |       |   |   |   \-- ✅ test_weight_boosting.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _bagging.py
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _forest.py
|   |   |       |   |   |-- ✅ _gb.py
|   |   |       |   |   |-- ✅ _gradient_boosting.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _gradient_boosting.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _gradient_boosting.pyx
|   |   |       |   |   |-- ✅ _iforest.py
|   |   |       |   |   |-- ✅ _stacking.py
|   |   |       |   |   |-- ✅ _voting.py
|   |   |       |   |   |-- ✅ _weight_boosting.py
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ experimental/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_enable_hist_gradient_boosting.py
|   |   |       |   |   |   |-- ✅ test_enable_iterative_imputer.py
|   |   |       |   |   |   \-- ✅ test_enable_successive_halving.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ enable_halving_search_cv.py
|   |   |       |   |   |-- ✅ enable_hist_gradient_boosting.py
|   |   |       |   |   \-- ✅ enable_iterative_imputer.py
|   |   |       |   |-- ✅ externals/
|   |   |       |   |   |-- ✅ _packaging/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _structures.py
|   |   |       |   |   |   \-- ✅ version.py
|   |   |       |   |   |-- ✅ _scipy/
|   |   |       |   |   |   |-- ✅ sparse/
|   |   |       |   |   |   |   |-- ✅ csgraph/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   \-- ✅ _laplacian.py
|   |   |       |   |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |-- ✅ array_api_compat/
|   |   |       |   |   |   |-- ✅ common/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _aliases.py
|   |   |       |   |   |   |   |-- ✅ _fft.py
|   |   |       |   |   |   |   |-- ✅ _helpers.py
|   |   |       |   |   |   |   |-- ✅ _linalg.py
|   |   |       |   |   |   |   \-- ✅ _typing.py
|   |   |       |   |   |   |-- ✅ cupy/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _aliases.py
|   |   |       |   |   |   |   |-- ✅ _info.py
|   |   |       |   |   |   |   |-- ✅ _typing.py
|   |   |       |   |   |   |   |-- ✅ fft.py
|   |   |       |   |   |   |   \-- ✅ linalg.py
|   |   |       |   |   |   |-- ✅ dask/
|   |   |       |   |   |   |   |-- ✅ array/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ _aliases.py
|   |   |       |   |   |   |   |   |-- ✅ _info.py
|   |   |       |   |   |   |   |   |-- ✅ fft.py
|   |   |       |   |   |   |   |   \-- ✅ linalg.py
|   |   |       |   |   |   |   \-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ numpy/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _aliases.py
|   |   |       |   |   |   |   |-- ✅ _info.py
|   |   |       |   |   |   |   |-- ✅ _typing.py
|   |   |       |   |   |   |   |-- ✅ fft.py
|   |   |       |   |   |   |   \-- ✅ linalg.py
|   |   |       |   |   |   |-- ✅ torch/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _aliases.py
|   |   |       |   |   |   |   |-- ✅ _info.py
|   |   |       |   |   |   |   |-- ✅ _typing.py
|   |   |       |   |   |   |   |-- ✅ fft.py
|   |   |       |   |   |   |   \-- ✅ linalg.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _internal.py
|   |   |       |   |   |   |-- ✅ LICENSE
|   |   |       |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   \-- ✅ README.md
|   |   |       |   |   |-- ✅ array_api_extra/
|   |   |       |   |   |   |-- ✅ _lib/
|   |   |       |   |   |   |   |-- ✅ _utils/
|   |   |       |   |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |   |-- ✅ _compat.py
|   |   |       |   |   |   |   |   |-- ✅ _compat.pyi
|   |   |       |   |   |   |   |   |-- ✅ _helpers.py
|   |   |       |   |   |   |   |   |-- ✅ _typing.py
|   |   |       |   |   |   |   |   \-- ✅ _typing.pyi
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ _at.py
|   |   |       |   |   |   |   |-- ✅ _backends.py
|   |   |       |   |   |   |   |-- ✅ _funcs.py
|   |   |       |   |   |   |   |-- ✅ _lazy.py
|   |   |       |   |   |   |   \-- ✅ _testing.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _delegation.py
|   |   |       |   |   |   |-- ✅ LICENSE
|   |   |       |   |   |   |-- ✅ py.typed
|   |   |       |   |   |   |-- ✅ README.md
|   |   |       |   |   |   \-- ✅ testing.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _arff.py
|   |   |       |   |   |-- ✅ _array_api_compat_vendor.py
|   |   |       |   |   |-- ✅ conftest.py
|   |   |       |   |   \-- ✅ README
|   |   |       |   |-- ✅ feature_extraction/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_dict_vectorizer.py
|   |   |       |   |   |   |-- ✅ test_feature_hasher.py
|   |   |       |   |   |   |-- ✅ test_image.py
|   |   |       |   |   |   \-- ✅ test_text.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _dict_vectorizer.py
|   |   |       |   |   |-- ✅ _hash.py
|   |   |       |   |   |-- ✅ _hashing_fast.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _hashing_fast.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _hashing_fast.pyx
|   |   |       |   |   |-- ✅ _stop_words.py
|   |   |       |   |   |-- ✅ image.py
|   |   |       |   |   |-- ✅ meson.build
|   |   |       |   |   \-- ✅ text.py
|   |   |       |   |-- ✅ feature_selection/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_base.py
|   |   |       |   |   |   |-- ✅ test_chi2.py
|   |   |       |   |   |   |-- ✅ test_feature_select.py
|   |   |       |   |   |   |-- ✅ test_from_model.py
|   |   |       |   |   |   |-- ✅ test_mutual_info.py
|   |   |       |   |   |   |-- ✅ test_rfe.py
|   |   |       |   |   |   |-- ✅ test_sequential.py
|   |   |       |   |   |   \-- ✅ test_variance_threshold.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _from_model.py
|   |   |       |   |   |-- ✅ _mutual_info.py
|   |   |       |   |   |-- ✅ _rfe.py
|   |   |       |   |   |-- ✅ _sequential.py
|   |   |       |   |   |-- ✅ _univariate_selection.py
|   |   |       |   |   \-- ✅ _variance_threshold.py
|   |   |       |   |-- ✅ frozen/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ test_frozen.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ _frozen.py
|   |   |       |   |-- ✅ gaussian_process/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _mini_sequence_kernel.py
|   |   |       |   |   |   |-- ✅ test_gpc.py
|   |   |       |   |   |   |-- ✅ test_gpr.py
|   |   |       |   |   |   \-- ✅ test_kernels.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _gpc.py
|   |   |       |   |   |-- ✅ _gpr.py
|   |   |       |   |   \-- ✅ kernels.py
|   |   |       |   |-- ✅ impute/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_base.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_impute.py
|   |   |       |   |   |   \-- ✅ test_knn.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _iterative.py
|   |   |       |   |   \-- ✅ _knn.py
|   |   |       |   |-- ✅ inspection/
|   |   |       |   |   |-- ✅ _plot/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_boundary_decision_display.py
|   |   |       |   |   |   |   \-- ✅ test_plot_partial_dependence.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ decision_boundary.py
|   |   |       |   |   |   \-- ✅ partial_dependence.py
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_partial_dependence.py
|   |   |       |   |   |   |-- ✅ test_pd_utils.py
|   |   |       |   |   |   \-- ✅ test_permutation_importance.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _partial_dependence.py
|   |   |       |   |   |-- ✅ _pd_utils.py
|   |   |       |   |   \-- ✅ _permutation_importance.py
|   |   |       |   |-- ✅ linear_model/
|   |   |       |   |   |-- ✅ _glm/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   \-- ✅ test_glm.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _newton_solver.py
|   |   |       |   |   |   \-- ✅ glm.py
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_base.py
|   |   |       |   |   |   |-- ✅ test_bayes.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_coordinate_descent.py
|   |   |       |   |   |   |-- ✅ test_huber.py
|   |   |       |   |   |   |-- ✅ test_least_angle.py
|   |   |       |   |   |   |-- ✅ test_linear_loss.py
|   |   |       |   |   |   |-- ✅ test_logistic.py
|   |   |       |   |   |   |-- ✅ test_omp.py
|   |   |       |   |   |   |-- ✅ test_passive_aggressive.py
|   |   |       |   |   |   |-- ✅ test_perceptron.py
|   |   |       |   |   |   |-- ✅ test_quantile.py
|   |   |       |   |   |   |-- ✅ test_ransac.py
|   |   |       |   |   |   |-- ✅ test_ridge.py
|   |   |       |   |   |   |-- ✅ test_sag.py
|   |   |       |   |   |   |-- ✅ test_sgd.py
|   |   |       |   |   |   |-- ✅ test_sparse_coordinate_descent.py
|   |   |       |   |   |   \-- ✅ test_theil_sen.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _bayes.py
|   |   |       |   |   |-- ✅ _cd_fast.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _cd_fast.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _cd_fast.pyx
|   |   |       |   |   |-- ✅ _coordinate_descent.py
|   |   |       |   |   |-- ✅ _huber.py
|   |   |       |   |   |-- ✅ _least_angle.py
|   |   |       |   |   |-- ✅ _linear_loss.py
|   |   |       |   |   |-- ✅ _logistic.py
|   |   |       |   |   |-- ✅ _omp.py
|   |   |       |   |   |-- ✅ _passive_aggressive.py
|   |   |       |   |   |-- ✅ _perceptron.py
|   |   |       |   |   |-- ✅ _quantile.py
|   |   |       |   |   |-- ✅ _ransac.py
|   |   |       |   |   |-- ✅ _ridge.py
|   |   |       |   |   |-- ✅ _sag.py
|   |   |       |   |   |-- ✅ _sag_fast.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _sag_fast.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _sag_fast.pyx.tp
|   |   |       |   |   |-- ✅ _sgd_fast.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _sgd_fast.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _sgd_fast.pyx.tp
|   |   |       |   |   |-- ✅ _stochastic_gradient.py
|   |   |       |   |   |-- ✅ _theil_sen.py
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ manifold/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_isomap.py
|   |   |       |   |   |   |-- ✅ test_locally_linear.py
|   |   |       |   |   |   |-- ✅ test_mds.py
|   |   |       |   |   |   |-- ✅ test_spectral_embedding.py
|   |   |       |   |   |   \-- ✅ test_t_sne.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _barnes_hut_tsne.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _barnes_hut_tsne.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _barnes_hut_tsne.pyx
|   |   |       |   |   |-- ✅ _isomap.py
|   |   |       |   |   |-- ✅ _locally_linear.py
|   |   |       |   |   |-- ✅ _mds.py
|   |   |       |   |   |-- ✅ _spectral_embedding.py
|   |   |       |   |   |-- ✅ _t_sne.py
|   |   |       |   |   |-- ✅ _utils.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _utils.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _utils.pyx
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ metrics/
|   |   |       |   |   |-- ✅ _pairwise_distances_reduction/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _argkmin.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _argkmin.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _argkmin.pxd.tp
|   |   |       |   |   |   |-- ✅ _argkmin.pyx.tp
|   |   |       |   |   |   |-- ✅ _argkmin_classmode.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _argkmin_classmode.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _argkmin_classmode.pyx.tp
|   |   |       |   |   |   |-- ✅ _base.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _base.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _base.pxd.tp
|   |   |       |   |   |   |-- ✅ _base.pyx.tp
|   |   |       |   |   |   |-- ✅ _classmode.pxd
|   |   |       |   |   |   |-- ✅ _datasets_pair.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _datasets_pair.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _datasets_pair.pxd.tp
|   |   |       |   |   |   |-- ✅ _datasets_pair.pyx.tp
|   |   |       |   |   |   |-- ✅ _dispatcher.py
|   |   |       |   |   |   |-- ✅ _middle_term_computer.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _middle_term_computer.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _middle_term_computer.pxd.tp
|   |   |       |   |   |   |-- ✅ _middle_term_computer.pyx.tp
|   |   |       |   |   |   |-- ✅ _radius_neighbors.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _radius_neighbors.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _radius_neighbors.pxd.tp
|   |   |       |   |   |   |-- ✅ _radius_neighbors.pyx.tp
|   |   |       |   |   |   |-- ✅ _radius_neighbors_classmode.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _radius_neighbors_classmode.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _radius_neighbors_classmode.pyx.tp
|   |   |       |   |   |   \-- ✅ meson.build
|   |   |       |   |   |-- ✅ _plot/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_common_curve_display.py
|   |   |       |   |   |   |   |-- ✅ test_confusion_matrix_display.py
|   |   |       |   |   |   |   |-- ✅ test_det_curve_display.py
|   |   |       |   |   |   |   |-- ✅ test_precision_recall_display.py
|   |   |       |   |   |   |   |-- ✅ test_predict_error_display.py
|   |   |       |   |   |   |   \-- ✅ test_roc_curve_display.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ confusion_matrix.py
|   |   |       |   |   |   |-- ✅ det_curve.py
|   |   |       |   |   |   |-- ✅ precision_recall_curve.py
|   |   |       |   |   |   |-- ✅ regression.py
|   |   |       |   |   |   \-- ✅ roc_curve.py
|   |   |       |   |   |-- ✅ cluster/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_bicluster.py
|   |   |       |   |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |   |-- ✅ test_supervised.py
|   |   |       |   |   |   |   \-- ✅ test_unsupervised.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _bicluster.py
|   |   |       |   |   |   |-- ✅ _expected_mutual_info_fast.cp312-win_amd64.lib
|   |   |       |   |   |   |-- ✅ _expected_mutual_info_fast.cp312-win_amd64.pyd
|   |   |       |   |   |   |-- ✅ _expected_mutual_info_fast.pyx
|   |   |       |   |   |   |-- ✅ _supervised.py
|   |   |       |   |   |   |-- ✅ _unsupervised.py
|   |   |       |   |   |   \-- ✅ meson.build
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_classification.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_dist_metrics.py
|   |   |       |   |   |   |-- ✅ test_pairwise.py
|   |   |       |   |   |   |-- ✅ test_pairwise_distances_reduction.py
|   |   |       |   |   |   |-- ✅ test_ranking.py
|   |   |       |   |   |   |-- ✅ test_regression.py
|   |   |       |   |   |   \-- ✅ test_score_objects.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _classification.py
|   |   |       |   |   |-- ✅ _dist_metrics.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _dist_metrics.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _dist_metrics.pxd
|   |   |       |   |   |-- ✅ _dist_metrics.pxd.tp
|   |   |       |   |   |-- ✅ _dist_metrics.pyx.tp
|   |   |       |   |   |-- ✅ _pairwise_fast.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _pairwise_fast.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _pairwise_fast.pyx
|   |   |       |   |   |-- ✅ _ranking.py
|   |   |       |   |   |-- ✅ _regression.py
|   |   |       |   |   |-- ✅ _scorer.py
|   |   |       |   |   |-- ✅ meson.build
|   |   |       |   |   \-- ✅ pairwise.py
|   |   |       |   |-- ✅ mixture/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_bayesian_mixture.py
|   |   |       |   |   |   |-- ✅ test_gaussian_mixture.py
|   |   |       |   |   |   \-- ✅ test_mixture.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _bayesian_mixture.py
|   |   |       |   |   \-- ✅ _gaussian_mixture.py
|   |   |       |   |-- ✅ model_selection/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ common.py
|   |   |       |   |   |   |-- ✅ test_classification_threshold.py
|   |   |       |   |   |   |-- ✅ test_plot.py
|   |   |       |   |   |   |-- ✅ test_search.py
|   |   |       |   |   |   |-- ✅ test_split.py
|   |   |       |   |   |   |-- ✅ test_successive_halving.py
|   |   |       |   |   |   \-- ✅ test_validation.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _classification_threshold.py
|   |   |       |   |   |-- ✅ _plot.py
|   |   |       |   |   |-- ✅ _search.py
|   |   |       |   |   |-- ✅ _search_successive_halving.py
|   |   |       |   |   |-- ✅ _split.py
|   |   |       |   |   \-- ✅ _validation.py
|   |   |       |   |-- ✅ neighbors/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_ball_tree.py
|   |   |       |   |   |   |-- ✅ test_graph.py
|   |   |       |   |   |   |-- ✅ test_kd_tree.py
|   |   |       |   |   |   |-- ✅ test_kde.py
|   |   |       |   |   |   |-- ✅ test_lof.py
|   |   |       |   |   |   |-- ✅ test_nca.py
|   |   |       |   |   |   |-- ✅ test_nearest_centroid.py
|   |   |       |   |   |   |-- ✅ test_neighbors.py
|   |   |       |   |   |   |-- ✅ test_neighbors_pipeline.py
|   |   |       |   |   |   |-- ✅ test_neighbors_tree.py
|   |   |       |   |   |   \-- ✅ test_quad_tree.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _ball_tree.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _ball_tree.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _ball_tree.pyx.tp
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _binary_tree.pxi.tp
|   |   |       |   |   |-- ✅ _classification.py
|   |   |       |   |   |-- ✅ _graph.py
|   |   |       |   |   |-- ✅ _kd_tree.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _kd_tree.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _kd_tree.pyx.tp
|   |   |       |   |   |-- ✅ _kde.py
|   |   |       |   |   |-- ✅ _lof.py
|   |   |       |   |   |-- ✅ _nca.py
|   |   |       |   |   |-- ✅ _nearest_centroid.py
|   |   |       |   |   |-- ✅ _partition_nodes.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _partition_nodes.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _partition_nodes.pxd
|   |   |       |   |   |-- ✅ _partition_nodes.pyx
|   |   |       |   |   |-- ✅ _quad_tree.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _quad_tree.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _quad_tree.pxd
|   |   |       |   |   |-- ✅ _quad_tree.pyx
|   |   |       |   |   |-- ✅ _regression.py
|   |   |       |   |   |-- ✅ _unsupervised.py
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ neural_network/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_base.py
|   |   |       |   |   |   |-- ✅ test_mlp.py
|   |   |       |   |   |   |-- ✅ test_rbm.py
|   |   |       |   |   |   \-- ✅ test_stochastic_optimizers.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _multilayer_perceptron.py
|   |   |       |   |   |-- ✅ _rbm.py
|   |   |       |   |   \-- ✅ _stochastic_optimizers.py
|   |   |       |   |-- ✅ preprocessing/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_common.py
|   |   |       |   |   |   |-- ✅ test_data.py
|   |   |       |   |   |   |-- ✅ test_discretization.py
|   |   |       |   |   |   |-- ✅ test_encoders.py
|   |   |       |   |   |   |-- ✅ test_function_transformer.py
|   |   |       |   |   |   |-- ✅ test_label.py
|   |   |       |   |   |   |-- ✅ test_polynomial.py
|   |   |       |   |   |   \-- ✅ test_target_encoder.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _csr_polynomial_expansion.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _csr_polynomial_expansion.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _csr_polynomial_expansion.pyx
|   |   |       |   |   |-- ✅ _data.py
|   |   |       |   |   |-- ✅ _discretization.py
|   |   |       |   |   |-- ✅ _encoders.py
|   |   |       |   |   |-- ✅ _function_transformer.py
|   |   |       |   |   |-- ✅ _label.py
|   |   |       |   |   |-- ✅ _polynomial.py
|   |   |       |   |   |-- ✅ _target_encoder.py
|   |   |       |   |   |-- ✅ _target_encoder_fast.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _target_encoder_fast.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _target_encoder_fast.pyx
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ semi_supervised/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_label_propagation.py
|   |   |       |   |   |   \-- ✅ test_self_training.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _label_propagation.py
|   |   |       |   |   \-- ✅ _self_training.py
|   |   |       |   |-- ✅ svm/
|   |   |       |   |   |-- ✅ src/
|   |   |       |   |   |   |-- ✅ liblinear/
|   |   |       |   |   |   |   |-- ✅ _cython_blas_helpers.h
|   |   |       |   |   |   |   |-- ✅ COPYRIGHT
|   |   |       |   |   |   |   |-- ✅ liblinear_helper.c
|   |   |       |   |   |   |   |-- ✅ linear.cpp
|   |   |       |   |   |   |   |-- ✅ linear.h
|   |   |       |   |   |   |   |-- ✅ tron.cpp
|   |   |       |   |   |   |   \-- ✅ tron.h
|   |   |       |   |   |   |-- ✅ libsvm/
|   |   |       |   |   |   |   |-- ✅ _svm_cython_blas_helpers.h
|   |   |       |   |   |   |   |-- ✅ LIBSVM_CHANGES
|   |   |       |   |   |   |   |-- ✅ libsvm_helper.c
|   |   |       |   |   |   |   |-- ✅ libsvm_sparse_helper.c
|   |   |       |   |   |   |   |-- ✅ libsvm_template.cpp
|   |   |       |   |   |   |   |-- ✅ svm.cpp
|   |   |       |   |   |   |   \-- ✅ svm.h
|   |   |       |   |   |   \-- ✅ newrand/
|   |   |       |   |   |       \-- ✅ newrand.h
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_bounds.py
|   |   |       |   |   |   |-- ✅ test_sparse.py
|   |   |       |   |   |   \-- ✅ test_svm.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _base.py
|   |   |       |   |   |-- ✅ _bounds.py
|   |   |       |   |   |-- ✅ _classes.py
|   |   |       |   |   |-- ✅ _liblinear.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _liblinear.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _liblinear.pxi
|   |   |       |   |   |-- ✅ _liblinear.pyx
|   |   |       |   |   |-- ✅ _libsvm.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _libsvm.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _libsvm.pxi
|   |   |       |   |   |-- ✅ _libsvm.pyx
|   |   |       |   |   |-- ✅ _libsvm_sparse.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _libsvm_sparse.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _libsvm_sparse.pyx
|   |   |       |   |   |-- ✅ _newrand.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _newrand.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _newrand.pyx
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ tests/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ metadata_routing_common.py
|   |   |       |   |   |-- ✅ test_base.py
|   |   |       |   |   |-- ✅ test_build.py
|   |   |       |   |   |-- ✅ test_calibration.py
|   |   |       |   |   |-- ✅ test_check_build.py
|   |   |       |   |   |-- ✅ test_common.py
|   |   |       |   |   |-- ✅ test_config.py
|   |   |       |   |   |-- ✅ test_discriminant_analysis.py
|   |   |       |   |   |-- ✅ test_docstring_parameters.py
|   |   |       |   |   |-- ✅ test_docstring_parameters_consistency.py
|   |   |       |   |   |-- ✅ test_docstrings.py
|   |   |       |   |   |-- ✅ test_dummy.py
|   |   |       |   |   |-- ✅ test_init.py
|   |   |       |   |   |-- ✅ test_isotonic.py
|   |   |       |   |   |-- ✅ test_kernel_approximation.py
|   |   |       |   |   |-- ✅ test_kernel_ridge.py
|   |   |       |   |   |-- ✅ test_metadata_routing.py
|   |   |       |   |   |-- ✅ test_metaestimators.py
|   |   |       |   |   |-- ✅ test_metaestimators_metadata_routing.py
|   |   |       |   |   |-- ✅ test_min_dependencies_readme.py
|   |   |       |   |   |-- ✅ test_multiclass.py
|   |   |       |   |   |-- ✅ test_multioutput.py
|   |   |       |   |   |-- ✅ test_naive_bayes.py
|   |   |       |   |   |-- ✅ test_pipeline.py
|   |   |       |   |   |-- ✅ test_public_functions.py
|   |   |       |   |   \-- ✅ test_random_projection.py
|   |   |       |   |-- ✅ tree/
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_export.py
|   |   |       |   |   |   |-- ✅ test_monotonic_tree.py
|   |   |       |   |   |   |-- ✅ test_reingold_tilford.py
|   |   |       |   |   |   \-- ✅ test_tree.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _classes.py
|   |   |       |   |   |-- ✅ _criterion.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _criterion.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _criterion.pxd
|   |   |       |   |   |-- ✅ _criterion.pyx
|   |   |       |   |   |-- ✅ _export.py
|   |   |       |   |   |-- ✅ _partitioner.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _partitioner.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _partitioner.pxd
|   |   |       |   |   |-- ✅ _partitioner.pyx
|   |   |       |   |   |-- ✅ _reingold_tilford.py
|   |   |       |   |   |-- ✅ _splitter.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _splitter.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _splitter.pxd
|   |   |       |   |   |-- ✅ _splitter.pyx
|   |   |       |   |   |-- ✅ _tree.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _tree.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _tree.pxd
|   |   |       |   |   |-- ✅ _tree.pyx
|   |   |       |   |   |-- ✅ _utils.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _utils.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _utils.pxd
|   |   |       |   |   |-- ✅ _utils.pyx
|   |   |       |   |   \-- ✅ meson.build
|   |   |       |   |-- ✅ utils/
|   |   |       |   |   |-- ✅ _repr_html/
|   |   |       |   |   |   |-- ✅ tests/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ test_estimator.py
|   |   |       |   |   |   |   \-- ✅ test_params.py
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ estimator.css
|   |   |       |   |   |   |-- ✅ estimator.js
|   |   |       |   |   |   |-- ✅ estimator.py
|   |   |       |   |   |   |-- ✅ params.css
|   |   |       |   |   |   \-- ✅ params.py
|   |   |       |   |   |-- ✅ _test_common/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ instance_generator.py
|   |   |       |   |   |-- ✅ src/
|   |   |       |   |   |   |-- ✅ MurmurHash3.cpp
|   |   |       |   |   |   \-- ✅ MurmurHash3.h
|   |   |       |   |   |-- ✅ tests/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_arpack.py
|   |   |       |   |   |   |-- ✅ test_array_api.py
|   |   |       |   |   |   |-- ✅ test_arrayfuncs.py
|   |   |       |   |   |   |-- ✅ test_bunch.py
|   |   |       |   |   |   |-- ✅ test_chunking.py
|   |   |       |   |   |   |-- ✅ test_class_weight.py
|   |   |       |   |   |   |-- ✅ test_cython_blas.py
|   |   |       |   |   |   |-- ✅ test_deprecation.py
|   |   |       |   |   |   |-- ✅ test_encode.py
|   |   |       |   |   |   |-- ✅ test_estimator_checks.py
|   |   |       |   |   |   |-- ✅ test_estimator_html_repr.py
|   |   |       |   |   |   |-- ✅ test_extmath.py
|   |   |       |   |   |   |-- ✅ test_fast_dict.py
|   |   |       |   |   |   |-- ✅ test_fixes.py
|   |   |       |   |   |   |-- ✅ test_graph.py
|   |   |       |   |   |   |-- ✅ test_indexing.py
|   |   |       |   |   |   |-- ✅ test_mask.py
|   |   |       |   |   |   |-- ✅ test_metaestimators.py
|   |   |       |   |   |   |-- ✅ test_missing.py
|   |   |       |   |   |   |-- ✅ test_mocking.py
|   |   |       |   |   |   |-- ✅ test_multiclass.py
|   |   |       |   |   |   |-- ✅ test_murmurhash.py
|   |   |       |   |   |   |-- ✅ test_optimize.py
|   |   |       |   |   |   |-- ✅ test_parallel.py
|   |   |       |   |   |   |-- ✅ test_param_validation.py
|   |   |       |   |   |   |-- ✅ test_plotting.py
|   |   |       |   |   |   |-- ✅ test_pprint.py
|   |   |       |   |   |   |-- ✅ test_random.py
|   |   |       |   |   |   |-- ✅ test_response.py
|   |   |       |   |   |   |-- ✅ test_seq_dataset.py
|   |   |       |   |   |   |-- ✅ test_set_output.py
|   |   |       |   |   |   |-- ✅ test_shortest_path.py
|   |   |       |   |   |   |-- ✅ test_show_versions.py
|   |   |       |   |   |   |-- ✅ test_sparsefuncs.py
|   |   |       |   |   |   |-- ✅ test_stats.py
|   |   |       |   |   |   |-- ✅ test_tags.py
|   |   |       |   |   |   |-- ✅ test_testing.py
|   |   |       |   |   |   |-- ✅ test_typedefs.py
|   |   |       |   |   |   |-- ✅ test_unique.py
|   |   |       |   |   |   |-- ✅ test_user_interface.py
|   |   |       |   |   |   |-- ✅ test_validation.py
|   |   |       |   |   |   \-- ✅ test_weight_vector.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _arpack.py
|   |   |       |   |   |-- ✅ _array_api.py
|   |   |       |   |   |-- ✅ _available_if.py
|   |   |       |   |   |-- ✅ _bunch.py
|   |   |       |   |   |-- ✅ _chunking.py
|   |   |       |   |   |-- ✅ _cython_blas.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _cython_blas.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _cython_blas.pxd
|   |   |       |   |   |-- ✅ _cython_blas.pyx
|   |   |       |   |   |-- ✅ _encode.py
|   |   |       |   |   |-- ✅ _estimator_html_repr.py
|   |   |       |   |   |-- ✅ _fast_dict.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _fast_dict.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _fast_dict.pxd
|   |   |       |   |   |-- ✅ _fast_dict.pyx
|   |   |       |   |   |-- ✅ _heap.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _heap.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _heap.pxd
|   |   |       |   |   |-- ✅ _heap.pyx
|   |   |       |   |   |-- ✅ _indexing.py
|   |   |       |   |   |-- ✅ _isfinite.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _isfinite.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _isfinite.pyx
|   |   |       |   |   |-- ✅ _mask.py
|   |   |       |   |   |-- ✅ _metadata_requests.py
|   |   |       |   |   |-- ✅ _missing.py
|   |   |       |   |   |-- ✅ _mocking.py
|   |   |       |   |   |-- ✅ _openmp_helpers.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _openmp_helpers.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _openmp_helpers.pxd
|   |   |       |   |   |-- ✅ _openmp_helpers.pyx
|   |   |       |   |   |-- ✅ _optional_dependencies.py
|   |   |       |   |   |-- ✅ _param_validation.py
|   |   |       |   |   |-- ✅ _plotting.py
|   |   |       |   |   |-- ✅ _pprint.py
|   |   |       |   |   |-- ✅ _random.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _random.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _random.pxd
|   |   |       |   |   |-- ✅ _random.pyx
|   |   |       |   |   |-- ✅ _response.py
|   |   |       |   |   |-- ✅ _seq_dataset.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _seq_dataset.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _seq_dataset.pxd.tp
|   |   |       |   |   |-- ✅ _seq_dataset.pyx.tp
|   |   |       |   |   |-- ✅ _set_output.py
|   |   |       |   |   |-- ✅ _show_versions.py
|   |   |       |   |   |-- ✅ _sorting.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _sorting.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _sorting.pxd
|   |   |       |   |   |-- ✅ _sorting.pyx
|   |   |       |   |   |-- ✅ _tags.py
|   |   |       |   |   |-- ✅ _testing.py
|   |   |       |   |   |-- ✅ _typedefs.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _typedefs.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _typedefs.pxd
|   |   |       |   |   |-- ✅ _typedefs.pyx
|   |   |       |   |   |-- ✅ _unique.py
|   |   |       |   |   |-- ✅ _user_interface.py
|   |   |       |   |   |-- ✅ _vector_sentinel.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _vector_sentinel.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _vector_sentinel.pxd
|   |   |       |   |   |-- ✅ _vector_sentinel.pyx
|   |   |       |   |   |-- ✅ _weight_vector.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ _weight_vector.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ _weight_vector.pxd.tp
|   |   |       |   |   |-- ✅ _weight_vector.pyx.tp
|   |   |       |   |   |-- ✅ arrayfuncs.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ arrayfuncs.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ arrayfuncs.pyx
|   |   |       |   |   |-- ✅ class_weight.py
|   |   |       |   |   |-- ✅ deprecation.py
|   |   |       |   |   |-- ✅ discovery.py
|   |   |       |   |   |-- ✅ estimator_checks.py
|   |   |       |   |   |-- ✅ extmath.py
|   |   |       |   |   |-- ✅ fixes.py
|   |   |       |   |   |-- ✅ graph.py
|   |   |       |   |   |-- ✅ meson.build
|   |   |       |   |   |-- ✅ metadata_routing.py
|   |   |       |   |   |-- ✅ metaestimators.py
|   |   |       |   |   |-- ✅ multiclass.py
|   |   |       |   |   |-- ✅ murmurhash.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ murmurhash.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ murmurhash.pxd
|   |   |       |   |   |-- ✅ murmurhash.pyx
|   |   |       |   |   |-- ✅ optimize.py
|   |   |       |   |   |-- ✅ parallel.py
|   |   |       |   |   |-- ✅ random.py
|   |   |       |   |   |-- ✅ sparsefuncs.py
|   |   |       |   |   |-- ✅ sparsefuncs_fast.cp312-win_amd64.lib
|   |   |       |   |   |-- ✅ sparsefuncs_fast.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ sparsefuncs_fast.pyx
|   |   |       |   |   |-- ✅ stats.py
|   |   |       |   |   \-- ✅ validation.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _built_with_meson.py
|   |   |       |   |-- ✅ _config.py
|   |   |       |   |-- ✅ _cyutility.cp312-win_amd64.lib
|   |   |       |   |-- ✅ _cyutility.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _distributor_init.py
|   |   |       |   |-- ✅ _isotonic.cp312-win_amd64.lib
|   |   |       |   |-- ✅ _isotonic.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _isotonic.pyx
|   |   |       |   |-- ✅ _min_dependencies.py
|   |   |       |   |-- ✅ base.py
|   |   |       |   |-- ✅ calibration.py
|   |   |       |   |-- ✅ conftest.py
|   |   |       |   |-- ✅ discriminant_analysis.py
|   |   |       |   |-- ✅ dummy.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ isotonic.py
|   |   |       |   |-- ✅ kernel_approximation.py
|   |   |       |   |-- ✅ kernel_ridge.py
|   |   |       |   |-- ✅ meson.build
|   |   |       |   |-- ✅ multiclass.py
|   |   |       |   |-- ✅ multioutput.py
|   |   |       |   |-- ✅ naive_bayes.py
|   |   |       |   |-- ✅ pipeline.py
|   |   |       |   \-- ✅ random_projection.py
|   |   |       |-- ✅ sniffio/
|   |   |       |   |-- ✅ _tests/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ test_sniffio.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _impl.py
|   |   |       |   |-- ✅ _version.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ sniffio-1.3.1.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE
|   |   |       |   |-- ✅ LICENSE.APACHE2
|   |   |       |   |-- ✅ LICENSE.MIT
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ sqlalchemy/
|   |   |       |   |-- ✅ connectors/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ aioodbc.py
|   |   |       |   |   |-- ✅ asyncio.py
|   |   |       |   |   \-- ✅ pyodbc.py
|   |   |       |   |-- ✅ cyextension/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ collections.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ collections.pyx
|   |   |       |   |   |-- ✅ immutabledict.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ immutabledict.pxd
|   |   |       |   |   |-- ✅ immutabledict.pyx
|   |   |       |   |   |-- ✅ processors.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ processors.pyx
|   |   |       |   |   |-- ✅ resultproxy.cp312-win_amd64.pyd
|   |   |       |   |   |-- ✅ resultproxy.pyx
|   |   |       |   |   |-- ✅ util.cp312-win_amd64.pyd
|   |   |       |   |   \-- ✅ util.pyx
|   |   |       |   |-- ✅ dialects/
|   |   |       |   |   |-- ✅ mssql/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ aioodbc.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ information_schema.py
|   |   |       |   |   |   |-- ✅ json.py
|   |   |       |   |   |   |-- ✅ provision.py
|   |   |       |   |   |   |-- ✅ pymssql.py
|   |   |       |   |   |   \-- ✅ pyodbc.py
|   |   |       |   |   |-- ✅ mysql/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ aiomysql.py
|   |   |       |   |   |   |-- ✅ asyncmy.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ cymysql.py
|   |   |       |   |   |   |-- ✅ dml.py
|   |   |       |   |   |   |-- ✅ enumerated.py
|   |   |       |   |   |   |-- ✅ expression.py
|   |   |       |   |   |   |-- ✅ json.py
|   |   |       |   |   |   |-- ✅ mariadb.py
|   |   |       |   |   |   |-- ✅ mariadbconnector.py
|   |   |       |   |   |   |-- ✅ mysqlconnector.py
|   |   |       |   |   |   |-- ✅ mysqldb.py
|   |   |       |   |   |   |-- ✅ provision.py
|   |   |       |   |   |   |-- ✅ pymysql.py
|   |   |       |   |   |   |-- ✅ pyodbc.py
|   |   |       |   |   |   |-- ✅ reflection.py
|   |   |       |   |   |   |-- ✅ reserved_words.py
|   |   |       |   |   |   \-- ✅ types.py
|   |   |       |   |   |-- ✅ oracle/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ cx_oracle.py
|   |   |       |   |   |   |-- ✅ dictionary.py
|   |   |       |   |   |   |-- ✅ oracledb.py
|   |   |       |   |   |   |-- ✅ provision.py
|   |   |       |   |   |   |-- ✅ types.py
|   |   |       |   |   |   \-- ✅ vector.py
|   |   |       |   |   |-- ✅ postgresql/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ _psycopg_common.py
|   |   |       |   |   |   |-- ✅ array.py
|   |   |       |   |   |   |-- ✅ asyncpg.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ dml.py
|   |   |       |   |   |   |-- ✅ ext.py
|   |   |       |   |   |   |-- ✅ hstore.py
|   |   |       |   |   |   |-- ✅ json.py
|   |   |       |   |   |   |-- ✅ named_types.py
|   |   |       |   |   |   |-- ✅ operators.py
|   |   |       |   |   |   |-- ✅ pg8000.py
|   |   |       |   |   |   |-- ✅ pg_catalog.py
|   |   |       |   |   |   |-- ✅ provision.py
|   |   |       |   |   |   |-- ✅ psycopg.py
|   |   |       |   |   |   |-- ✅ psycopg2.py
|   |   |       |   |   |   |-- ✅ psycopg2cffi.py
|   |   |       |   |   |   |-- ✅ ranges.py
|   |   |       |   |   |   \-- ✅ types.py
|   |   |       |   |   |-- ✅ sqlite/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ aiosqlite.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ dml.py
|   |   |       |   |   |   |-- ✅ json.py
|   |   |       |   |   |   |-- ✅ provision.py
|   |   |       |   |   |   |-- ✅ pysqlcipher.py
|   |   |       |   |   |   \-- ✅ pysqlite.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _typing.py
|   |   |       |   |   \-- ✅ type_migration_guidelines.txt
|   |   |       |   |-- ✅ engine/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _py_processors.py
|   |   |       |   |   |-- ✅ _py_row.py
|   |   |       |   |   |-- ✅ _py_util.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ characteristics.py
|   |   |       |   |   |-- ✅ create.py
|   |   |       |   |   |-- ✅ cursor.py
|   |   |       |   |   |-- ✅ default.py
|   |   |       |   |   |-- ✅ events.py
|   |   |       |   |   |-- ✅ interfaces.py
|   |   |       |   |   |-- ✅ mock.py
|   |   |       |   |   |-- ✅ processors.py
|   |   |       |   |   |-- ✅ reflection.py
|   |   |       |   |   |-- ✅ result.py
|   |   |       |   |   |-- ✅ row.py
|   |   |       |   |   |-- ✅ strategies.py
|   |   |       |   |   |-- ✅ url.py
|   |   |       |   |   \-- ✅ util.py
|   |   |       |   |-- ✅ event/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ api.py
|   |   |       |   |   |-- ✅ attr.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ legacy.py
|   |   |       |   |   \-- ✅ registry.py
|   |   |       |   |-- ✅ ext/
|   |   |       |   |   |-- ✅ asyncio/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ engine.py
|   |   |       |   |   |   |-- ✅ exc.py
|   |   |       |   |   |   |-- ✅ result.py
|   |   |       |   |   |   |-- ✅ scoping.py
|   |   |       |   |   |   \-- ✅ session.py
|   |   |       |   |   |-- ✅ declarative/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ extensions.py
|   |   |       |   |   |-- ✅ mypy/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ apply.py
|   |   |       |   |   |   |-- ✅ decl_class.py
|   |   |       |   |   |   |-- ✅ infer.py
|   |   |       |   |   |   |-- ✅ names.py
|   |   |       |   |   |   |-- ✅ plugin.py
|   |   |       |   |   |   \-- ✅ util.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ associationproxy.py
|   |   |       |   |   |-- ✅ automap.py
|   |   |       |   |   |-- ✅ baked.py
|   |   |       |   |   |-- ✅ compiler.py
|   |   |       |   |   |-- ✅ horizontal_shard.py
|   |   |       |   |   |-- ✅ hybrid.py
|   |   |       |   |   |-- ✅ indexable.py
|   |   |       |   |   |-- ✅ instrumentation.py
|   |   |       |   |   |-- ✅ mutable.py
|   |   |       |   |   |-- ✅ orderinglist.py
|   |   |       |   |   \-- ✅ serializer.py
|   |   |       |   |-- ✅ future/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ engine.py
|   |   |       |   |-- ✅ orm/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _orm_constructors.py
|   |   |       |   |   |-- ✅ _typing.py
|   |   |       |   |   |-- ✅ attributes.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ bulk_persistence.py
|   |   |       |   |   |-- ✅ clsregistry.py
|   |   |       |   |   |-- ✅ collections.py
|   |   |       |   |   |-- ✅ context.py
|   |   |       |   |   |-- ✅ decl_api.py
|   |   |       |   |   |-- ✅ decl_base.py
|   |   |       |   |   |-- ✅ dependency.py
|   |   |       |   |   |-- ✅ descriptor_props.py
|   |   |       |   |   |-- ✅ dynamic.py
|   |   |       |   |   |-- ✅ evaluator.py
|   |   |       |   |   |-- ✅ events.py
|   |   |       |   |   |-- ✅ exc.py
|   |   |       |   |   |-- ✅ identity.py
|   |   |       |   |   |-- ✅ instrumentation.py
|   |   |       |   |   |-- ✅ interfaces.py
|   |   |       |   |   |-- ✅ loading.py
|   |   |       |   |   |-- ✅ mapped_collection.py
|   |   |       |   |   |-- ✅ mapper.py
|   |   |       |   |   |-- ✅ path_registry.py
|   |   |       |   |   |-- ✅ persistence.py
|   |   |       |   |   |-- ✅ properties.py
|   |   |       |   |   |-- ✅ query.py
|   |   |       |   |   |-- ✅ relationships.py
|   |   |       |   |   |-- ✅ scoping.py
|   |   |       |   |   |-- ✅ session.py
|   |   |       |   |   |-- ✅ state.py
|   |   |       |   |   |-- ✅ state_changes.py
|   |   |       |   |   |-- ✅ strategies.py
|   |   |       |   |   |-- ✅ strategy_options.py
|   |   |       |   |   |-- ✅ sync.py
|   |   |       |   |   |-- ✅ unitofwork.py
|   |   |       |   |   |-- ✅ util.py
|   |   |       |   |   \-- ✅ writeonly.py
|   |   |       |   |-- ✅ pool/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ events.py
|   |   |       |   |   \-- ✅ impl.py
|   |   |       |   |-- ✅ sql/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _dml_constructors.py
|   |   |       |   |   |-- ✅ _elements_constructors.py
|   |   |       |   |   |-- ✅ _orm_types.py
|   |   |       |   |   |-- ✅ _py_util.py
|   |   |       |   |   |-- ✅ _selectable_constructors.py
|   |   |       |   |   |-- ✅ _typing.py
|   |   |       |   |   |-- ✅ annotation.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ cache_key.py
|   |   |       |   |   |-- ✅ coercions.py
|   |   |       |   |   |-- ✅ compiler.py
|   |   |       |   |   |-- ✅ crud.py
|   |   |       |   |   |-- ✅ ddl.py
|   |   |       |   |   |-- ✅ default_comparator.py
|   |   |       |   |   |-- ✅ dml.py
|   |   |       |   |   |-- ✅ elements.py
|   |   |       |   |   |-- ✅ events.py
|   |   |       |   |   |-- ✅ expression.py
|   |   |       |   |   |-- ✅ functions.py
|   |   |       |   |   |-- ✅ lambdas.py
|   |   |       |   |   |-- ✅ naming.py
|   |   |       |   |   |-- ✅ operators.py
|   |   |       |   |   |-- ✅ roles.py
|   |   |       |   |   |-- ✅ schema.py
|   |   |       |   |   |-- ✅ selectable.py
|   |   |       |   |   |-- ✅ sqltypes.py
|   |   |       |   |   |-- ✅ traversals.py
|   |   |       |   |   |-- ✅ type_api.py
|   |   |       |   |   |-- ✅ util.py
|   |   |       |   |   \-- ✅ visitors.py
|   |   |       |   |-- ✅ testing/
|   |   |       |   |   |-- ✅ fixtures/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ base.py
|   |   |       |   |   |   |-- ✅ mypy.py
|   |   |       |   |   |   |-- ✅ orm.py
|   |   |       |   |   |   \-- ✅ sql.py
|   |   |       |   |   |-- ✅ plugin/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ bootstrap.py
|   |   |       |   |   |   |-- ✅ plugin_base.py
|   |   |       |   |   |   \-- ✅ pytestplugin.py
|   |   |       |   |   |-- ✅ suite/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ test_cte.py
|   |   |       |   |   |   |-- ✅ test_ddl.py
|   |   |       |   |   |   |-- ✅ test_deprecations.py
|   |   |       |   |   |   |-- ✅ test_dialect.py
|   |   |       |   |   |   |-- ✅ test_insert.py
|   |   |       |   |   |   |-- ✅ test_reflection.py
|   |   |       |   |   |   |-- ✅ test_results.py
|   |   |       |   |   |   |-- ✅ test_rowcount.py
|   |   |       |   |   |   |-- ✅ test_select.py
|   |   |       |   |   |   |-- ✅ test_sequence.py
|   |   |       |   |   |   |-- ✅ test_types.py
|   |   |       |   |   |   |-- ✅ test_unicode_ddl.py
|   |   |       |   |   |   \-- ✅ test_update_delete.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ assertions.py
|   |   |       |   |   |-- ✅ assertsql.py
|   |   |       |   |   |-- ✅ asyncio.py
|   |   |       |   |   |-- ✅ config.py
|   |   |       |   |   |-- ✅ engines.py
|   |   |       |   |   |-- ✅ entities.py
|   |   |       |   |   |-- ✅ exclusions.py
|   |   |       |   |   |-- ✅ pickleable.py
|   |   |       |   |   |-- ✅ profiling.py
|   |   |       |   |   |-- ✅ provision.py
|   |   |       |   |   |-- ✅ requirements.py
|   |   |       |   |   |-- ✅ schema.py
|   |   |       |   |   |-- ✅ util.py
|   |   |       |   |   \-- ✅ warnings.py
|   |   |       |   |-- ✅ util/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ _collections.py
|   |   |       |   |   |-- ✅ _concurrency_py3k.py
|   |   |       |   |   |-- ✅ _has_cy.py
|   |   |       |   |   |-- ✅ _py_collections.py
|   |   |       |   |   |-- ✅ compat.py
|   |   |       |   |   |-- ✅ concurrency.py
|   |   |       |   |   |-- ✅ deprecations.py
|   |   |       |   |   |-- ✅ langhelpers.py
|   |   |       |   |   |-- ✅ preloaded.py
|   |   |       |   |   |-- ✅ queue.py
|   |   |       |   |   |-- ✅ tool_support.py
|   |   |       |   |   |-- ✅ topological.py
|   |   |       |   |   \-- ✅ typing.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ events.py
|   |   |       |   |-- ✅ exc.py
|   |   |       |   |-- ✅ inspection.py
|   |   |       |   |-- ✅ log.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ schema.py
|   |   |       |   \-- ✅ types.py
|   |   |       |-- ✅ sqlalchemy-2.0.43.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ starlette/
|   |   |       |   |-- ✅ middleware/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ authentication.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   |-- ✅ cors.py
|   |   |       |   |   |-- ✅ errors.py
|   |   |       |   |   |-- ✅ exceptions.py
|   |   |       |   |   |-- ✅ gzip.py
|   |   |       |   |   |-- ✅ httpsredirect.py
|   |   |       |   |   |-- ✅ sessions.py
|   |   |       |   |   |-- ✅ trustedhost.py
|   |   |       |   |   \-- ✅ wsgi.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _exception_handler.py
|   |   |       |   |-- ✅ _utils.py
|   |   |       |   |-- ✅ applications.py
|   |   |       |   |-- ✅ authentication.py
|   |   |       |   |-- ✅ background.py
|   |   |       |   |-- ✅ concurrency.py
|   |   |       |   |-- ✅ config.py
|   |   |       |   |-- ✅ convertors.py
|   |   |       |   |-- ✅ datastructures.py
|   |   |       |   |-- ✅ endpoints.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ formparsers.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ requests.py
|   |   |       |   |-- ✅ responses.py
|   |   |       |   |-- ✅ routing.py
|   |   |       |   |-- ✅ schemas.py
|   |   |       |   |-- ✅ staticfiles.py
|   |   |       |   |-- ✅ status.py
|   |   |       |   |-- ✅ templating.py
|   |   |       |   |-- ✅ testclient.py
|   |   |       |   |-- ✅ types.py
|   |   |       |   \-- ✅ websockets.py
|   |   |       |-- ✅ starlette-0.48.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE.md
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ threadpoolctl-3.6.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ typing_extensions-4.15.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ typing_inspection/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ introspection.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ typing_objects.py
|   |   |       |   \-- ✅ typing_objects.pyi
|   |   |       |-- ✅ typing_inspection-0.4.1.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ tzdata/
|   |   |       |   |-- ✅ zoneinfo/
|   |   |       |   |   |-- ✅ Africa/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Abidjan
|   |   |       |   |   |   |-- ✅ Accra
|   |   |       |   |   |   |-- ✅ Addis_Ababa
|   |   |       |   |   |   |-- ✅ Algiers
|   |   |       |   |   |   |-- ✅ Asmara
|   |   |       |   |   |   |-- ✅ Asmera
|   |   |       |   |   |   |-- ✅ Bamako
|   |   |       |   |   |   |-- ✅ Bangui
|   |   |       |   |   |   |-- ✅ Banjul
|   |   |       |   |   |   |-- ✅ Bissau
|   |   |       |   |   |   |-- ✅ Blantyre
|   |   |       |   |   |   |-- ✅ Brazzaville
|   |   |       |   |   |   |-- ✅ Bujumbura
|   |   |       |   |   |   |-- ✅ Cairo
|   |   |       |   |   |   |-- ✅ Casablanca
|   |   |       |   |   |   |-- ✅ Ceuta
|   |   |       |   |   |   |-- ✅ Conakry
|   |   |       |   |   |   |-- ✅ Dakar
|   |   |       |   |   |   |-- ✅ Dar_es_Salaam
|   |   |       |   |   |   |-- ✅ Djibouti
|   |   |       |   |   |   |-- ✅ Douala
|   |   |       |   |   |   |-- ✅ El_Aaiun
|   |   |       |   |   |   |-- ✅ Freetown
|   |   |       |   |   |   |-- ✅ Gaborone
|   |   |       |   |   |   |-- ✅ Harare
|   |   |       |   |   |   |-- ✅ Johannesburg
|   |   |       |   |   |   |-- ✅ Juba
|   |   |       |   |   |   |-- ✅ Kampala
|   |   |       |   |   |   |-- ✅ Khartoum
|   |   |       |   |   |   |-- ✅ Kigali
|   |   |       |   |   |   |-- ✅ Kinshasa
|   |   |       |   |   |   |-- ✅ Lagos
|   |   |       |   |   |   |-- ✅ Libreville
|   |   |       |   |   |   |-- ✅ Lome
|   |   |       |   |   |   |-- ✅ Luanda
|   |   |       |   |   |   |-- ✅ Lubumbashi
|   |   |       |   |   |   |-- ✅ Lusaka
|   |   |       |   |   |   |-- ✅ Malabo
|   |   |       |   |   |   |-- ✅ Maputo
|   |   |       |   |   |   |-- ✅ Maseru
|   |   |       |   |   |   |-- ✅ Mbabane
|   |   |       |   |   |   |-- ✅ Mogadishu
|   |   |       |   |   |   |-- ✅ Monrovia
|   |   |       |   |   |   |-- ✅ Nairobi
|   |   |       |   |   |   |-- ✅ Ndjamena
|   |   |       |   |   |   |-- ✅ Niamey
|   |   |       |   |   |   |-- ✅ Nouakchott
|   |   |       |   |   |   |-- ✅ Ouagadougou
|   |   |       |   |   |   |-- ✅ Porto-Novo
|   |   |       |   |   |   |-- ✅ Sao_Tome
|   |   |       |   |   |   |-- ✅ Timbuktu
|   |   |       |   |   |   |-- ✅ Tripoli
|   |   |       |   |   |   |-- ✅ Tunis
|   |   |       |   |   |   \-- ✅ Windhoek
|   |   |       |   |   |-- ✅ America/
|   |   |       |   |   |   |-- ✅ Argentina/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ Buenos_Aires
|   |   |       |   |   |   |   |-- ✅ Catamarca
|   |   |       |   |   |   |   |-- ✅ ComodRivadavia
|   |   |       |   |   |   |   |-- ✅ Cordoba
|   |   |       |   |   |   |   |-- ✅ Jujuy
|   |   |       |   |   |   |   |-- ✅ La_Rioja
|   |   |       |   |   |   |   |-- ✅ Mendoza
|   |   |       |   |   |   |   |-- ✅ Rio_Gallegos
|   |   |       |   |   |   |   |-- ✅ Salta
|   |   |       |   |   |   |   |-- ✅ San_Juan
|   |   |       |   |   |   |   |-- ✅ San_Luis
|   |   |       |   |   |   |   |-- ✅ Tucuman
|   |   |       |   |   |   |   \-- ✅ Ushuaia
|   |   |       |   |   |   |-- ✅ Indiana/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ Indianapolis
|   |   |       |   |   |   |   |-- ✅ Knox
|   |   |       |   |   |   |   |-- ✅ Marengo
|   |   |       |   |   |   |   |-- ✅ Petersburg
|   |   |       |   |   |   |   |-- ✅ Tell_City
|   |   |       |   |   |   |   |-- ✅ Vevay
|   |   |       |   |   |   |   |-- ✅ Vincennes
|   |   |       |   |   |   |   \-- ✅ Winamac
|   |   |       |   |   |   |-- ✅ Kentucky/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ Louisville
|   |   |       |   |   |   |   \-- ✅ Monticello
|   |   |       |   |   |   |-- ✅ North_Dakota/
|   |   |       |   |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |   |-- ✅ Beulah
|   |   |       |   |   |   |   |-- ✅ Center
|   |   |       |   |   |   |   \-- ✅ New_Salem
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Adak
|   |   |       |   |   |   |-- ✅ Anchorage
|   |   |       |   |   |   |-- ✅ Anguilla
|   |   |       |   |   |   |-- ✅ Antigua
|   |   |       |   |   |   |-- ✅ Araguaina
|   |   |       |   |   |   |-- ✅ Aruba
|   |   |       |   |   |   |-- ✅ Asuncion
|   |   |       |   |   |   |-- ✅ Atikokan
|   |   |       |   |   |   |-- ✅ Atka
|   |   |       |   |   |   |-- ✅ Bahia
|   |   |       |   |   |   |-- ✅ Bahia_Banderas
|   |   |       |   |   |   |-- ✅ Barbados
|   |   |       |   |   |   |-- ✅ Belem
|   |   |       |   |   |   |-- ✅ Belize
|   |   |       |   |   |   |-- ✅ Blanc-Sablon
|   |   |       |   |   |   |-- ✅ Boa_Vista
|   |   |       |   |   |   |-- ✅ Bogota
|   |   |       |   |   |   |-- ✅ Boise
|   |   |       |   |   |   |-- ✅ Buenos_Aires
|   |   |       |   |   |   |-- ✅ Cambridge_Bay
|   |   |       |   |   |   |-- ✅ Campo_Grande
|   |   |       |   |   |   |-- ✅ Cancun
|   |   |       |   |   |   |-- ✅ Caracas
|   |   |       |   |   |   |-- ✅ Catamarca
|   |   |       |   |   |   |-- ✅ Cayenne
|   |   |       |   |   |   |-- ✅ Cayman
|   |   |       |   |   |   |-- ✅ Chicago
|   |   |       |   |   |   |-- ✅ Chihuahua
|   |   |       |   |   |   |-- ✅ Ciudad_Juarez
|   |   |       |   |   |   |-- ✅ Coral_Harbour
|   |   |       |   |   |   |-- ✅ Cordoba
|   |   |       |   |   |   |-- ✅ Costa_Rica
|   |   |       |   |   |   |-- ✅ Coyhaique
|   |   |       |   |   |   |-- ✅ Creston
|   |   |       |   |   |   |-- ✅ Cuiaba
|   |   |       |   |   |   |-- ✅ Curacao
|   |   |       |   |   |   |-- ✅ Danmarkshavn
|   |   |       |   |   |   |-- ✅ Dawson
|   |   |       |   |   |   |-- ✅ Dawson_Creek
|   |   |       |   |   |   |-- ✅ Denver
|   |   |       |   |   |   |-- ✅ Detroit
|   |   |       |   |   |   |-- ✅ Dominica
|   |   |       |   |   |   |-- ✅ Edmonton
|   |   |       |   |   |   |-- ✅ Eirunepe
|   |   |       |   |   |   |-- ✅ El_Salvador
|   |   |       |   |   |   |-- ✅ Ensenada
|   |   |       |   |   |   |-- ✅ Fort_Nelson
|   |   |       |   |   |   |-- ✅ Fort_Wayne
|   |   |       |   |   |   |-- ✅ Fortaleza
|   |   |       |   |   |   |-- ✅ Glace_Bay
|   |   |       |   |   |   |-- ✅ Godthab
|   |   |       |   |   |   |-- ✅ Goose_Bay
|   |   |       |   |   |   |-- ✅ Grand_Turk
|   |   |       |   |   |   |-- ✅ Grenada
|   |   |       |   |   |   |-- ✅ Guadeloupe
|   |   |       |   |   |   |-- ✅ Guatemala
|   |   |       |   |   |   |-- ✅ Guayaquil
|   |   |       |   |   |   |-- ✅ Guyana
|   |   |       |   |   |   |-- ✅ Halifax
|   |   |       |   |   |   |-- ✅ Havana
|   |   |       |   |   |   |-- ✅ Hermosillo
|   |   |       |   |   |   |-- ✅ Indianapolis
|   |   |       |   |   |   |-- ✅ Inuvik
|   |   |       |   |   |   |-- ✅ Iqaluit
|   |   |       |   |   |   |-- ✅ Jamaica
|   |   |       |   |   |   |-- ✅ Jujuy
|   |   |       |   |   |   |-- ✅ Juneau
|   |   |       |   |   |   |-- ✅ Knox_IN
|   |   |       |   |   |   |-- ✅ Kralendijk
|   |   |       |   |   |   |-- ✅ La_Paz
|   |   |       |   |   |   |-- ✅ Lima
|   |   |       |   |   |   |-- ✅ Los_Angeles
|   |   |       |   |   |   |-- ✅ Louisville
|   |   |       |   |   |   |-- ✅ Lower_Princes
|   |   |       |   |   |   |-- ✅ Maceio
|   |   |       |   |   |   |-- ✅ Managua
|   |   |       |   |   |   |-- ✅ Manaus
|   |   |       |   |   |   |-- ✅ Marigot
|   |   |       |   |   |   |-- ✅ Martinique
|   |   |       |   |   |   |-- ✅ Matamoros
|   |   |       |   |   |   |-- ✅ Mazatlan
|   |   |       |   |   |   |-- ✅ Mendoza
|   |   |       |   |   |   |-- ✅ Menominee
|   |   |       |   |   |   |-- ✅ Merida
|   |   |       |   |   |   |-- ✅ Metlakatla
|   |   |       |   |   |   |-- ✅ Mexico_City
|   |   |       |   |   |   |-- ✅ Miquelon
|   |   |       |   |   |   |-- ✅ Moncton
|   |   |       |   |   |   |-- ✅ Monterrey
|   |   |       |   |   |   |-- ✅ Montevideo
|   |   |       |   |   |   |-- ✅ Montreal
|   |   |       |   |   |   |-- ✅ Montserrat
|   |   |       |   |   |   |-- ✅ Nassau
|   |   |       |   |   |   |-- ✅ New_York
|   |   |       |   |   |   |-- ✅ Nipigon
|   |   |       |   |   |   |-- ✅ Nome
|   |   |       |   |   |   |-- ✅ Noronha
|   |   |       |   |   |   |-- ✅ Nuuk
|   |   |       |   |   |   |-- ✅ Ojinaga
|   |   |       |   |   |   |-- ✅ Panama
|   |   |       |   |   |   |-- ✅ Pangnirtung
|   |   |       |   |   |   |-- ✅ Paramaribo
|   |   |       |   |   |   |-- ✅ Phoenix
|   |   |       |   |   |   |-- ✅ Port-au-Prince
|   |   |       |   |   |   |-- ✅ Port_of_Spain
|   |   |       |   |   |   |-- ✅ Porto_Acre
|   |   |       |   |   |   |-- ✅ Porto_Velho
|   |   |       |   |   |   |-- ✅ Puerto_Rico
|   |   |       |   |   |   |-- ✅ Punta_Arenas
|   |   |       |   |   |   |-- ✅ Rainy_River
|   |   |       |   |   |   |-- ✅ Rankin_Inlet
|   |   |       |   |   |   |-- ✅ Recife
|   |   |       |   |   |   |-- ✅ Regina
|   |   |       |   |   |   |-- ✅ Resolute
|   |   |       |   |   |   |-- ✅ Rio_Branco
|   |   |       |   |   |   |-- ✅ Rosario
|   |   |       |   |   |   |-- ✅ Santa_Isabel
|   |   |       |   |   |   |-- ✅ Santarem
|   |   |       |   |   |   |-- ✅ Santiago
|   |   |       |   |   |   |-- ✅ Santo_Domingo
|   |   |       |   |   |   |-- ✅ Sao_Paulo
|   |   |       |   |   |   |-- ✅ Scoresbysund
|   |   |       |   |   |   |-- ✅ Shiprock
|   |   |       |   |   |   |-- ✅ Sitka
|   |   |       |   |   |   |-- ✅ St_Barthelemy
|   |   |       |   |   |   |-- ✅ St_Johns
|   |   |       |   |   |   |-- ✅ St_Kitts
|   |   |       |   |   |   |-- ✅ St_Lucia
|   |   |       |   |   |   |-- ✅ St_Thomas
|   |   |       |   |   |   |-- ✅ St_Vincent
|   |   |       |   |   |   |-- ✅ Swift_Current
|   |   |       |   |   |   |-- ✅ Tegucigalpa
|   |   |       |   |   |   |-- ✅ Thule
|   |   |       |   |   |   |-- ✅ Thunder_Bay
|   |   |       |   |   |   |-- ✅ Tijuana
|   |   |       |   |   |   |-- ✅ Toronto
|   |   |       |   |   |   |-- ✅ Tortola
|   |   |       |   |   |   |-- ✅ Vancouver
|   |   |       |   |   |   |-- ✅ Virgin
|   |   |       |   |   |   |-- ✅ Whitehorse
|   |   |       |   |   |   |-- ✅ Winnipeg
|   |   |       |   |   |   |-- ✅ Yakutat
|   |   |       |   |   |   \-- ✅ Yellowknife
|   |   |       |   |   |-- ✅ Antarctica/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Casey
|   |   |       |   |   |   |-- ✅ Davis
|   |   |       |   |   |   |-- ✅ DumontDUrville
|   |   |       |   |   |   |-- ✅ Macquarie
|   |   |       |   |   |   |-- ✅ Mawson
|   |   |       |   |   |   |-- ✅ McMurdo
|   |   |       |   |   |   |-- ✅ Palmer
|   |   |       |   |   |   |-- ✅ Rothera
|   |   |       |   |   |   |-- ✅ South_Pole
|   |   |       |   |   |   |-- ✅ Syowa
|   |   |       |   |   |   |-- ✅ Troll
|   |   |       |   |   |   \-- ✅ Vostok
|   |   |       |   |   |-- ✅ Arctic/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   \-- ✅ Longyearbyen
|   |   |       |   |   |-- ✅ Asia/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Aden
|   |   |       |   |   |   |-- ✅ Almaty
|   |   |       |   |   |   |-- ✅ Amman
|   |   |       |   |   |   |-- ✅ Anadyr
|   |   |       |   |   |   |-- ✅ Aqtau
|   |   |       |   |   |   |-- ✅ Aqtobe
|   |   |       |   |   |   |-- ✅ Ashgabat
|   |   |       |   |   |   |-- ✅ Ashkhabad
|   |   |       |   |   |   |-- ✅ Atyrau
|   |   |       |   |   |   |-- ✅ Baghdad
|   |   |       |   |   |   |-- ✅ Bahrain
|   |   |       |   |   |   |-- ✅ Baku
|   |   |       |   |   |   |-- ✅ Bangkok
|   |   |       |   |   |   |-- ✅ Barnaul
|   |   |       |   |   |   |-- ✅ Beirut
|   |   |       |   |   |   |-- ✅ Bishkek
|   |   |       |   |   |   |-- ✅ Brunei
|   |   |       |   |   |   |-- ✅ Calcutta
|   |   |       |   |   |   |-- ✅ Chita
|   |   |       |   |   |   |-- ✅ Choibalsan
|   |   |       |   |   |   |-- ✅ Chongqing
|   |   |       |   |   |   |-- ✅ Chungking
|   |   |       |   |   |   |-- ✅ Colombo
|   |   |       |   |   |   |-- ✅ Dacca
|   |   |       |   |   |   |-- ✅ Damascus
|   |   |       |   |   |   |-- ✅ Dhaka
|   |   |       |   |   |   |-- ✅ Dili
|   |   |       |   |   |   |-- ✅ Dubai
|   |   |       |   |   |   |-- ✅ Dushanbe
|   |   |       |   |   |   |-- ✅ Famagusta
|   |   |       |   |   |   |-- ✅ Gaza
|   |   |       |   |   |   |-- ✅ Harbin
|   |   |       |   |   |   |-- ✅ Hebron
|   |   |       |   |   |   |-- ✅ Ho_Chi_Minh
|   |   |       |   |   |   |-- ✅ Hong_Kong
|   |   |       |   |   |   |-- ✅ Hovd
|   |   |       |   |   |   |-- ✅ Irkutsk
|   |   |       |   |   |   |-- ✅ Istanbul
|   |   |       |   |   |   |-- ✅ Jakarta
|   |   |       |   |   |   |-- ✅ Jayapura
|   |   |       |   |   |   |-- ✅ Jerusalem
|   |   |       |   |   |   |-- ✅ Kabul
|   |   |       |   |   |   |-- ✅ Kamchatka
|   |   |       |   |   |   |-- ✅ Karachi
|   |   |       |   |   |   |-- ✅ Kashgar
|   |   |       |   |   |   |-- ✅ Kathmandu
|   |   |       |   |   |   |-- ✅ Katmandu
|   |   |       |   |   |   |-- ✅ Khandyga
|   |   |       |   |   |   |-- ✅ Kolkata
|   |   |       |   |   |   |-- ✅ Krasnoyarsk
|   |   |       |   |   |   |-- ✅ Kuala_Lumpur
|   |   |       |   |   |   |-- ✅ Kuching
|   |   |       |   |   |   |-- ✅ Kuwait
|   |   |       |   |   |   |-- ✅ Macao
|   |   |       |   |   |   |-- ✅ Macau
|   |   |       |   |   |   |-- ✅ Magadan
|   |   |       |   |   |   |-- ✅ Makassar
|   |   |       |   |   |   |-- ✅ Manila
|   |   |       |   |   |   |-- ✅ Muscat
|   |   |       |   |   |   |-- ✅ Nicosia
|   |   |       |   |   |   |-- ✅ Novokuznetsk
|   |   |       |   |   |   |-- ✅ Novosibirsk
|   |   |       |   |   |   |-- ✅ Omsk
|   |   |       |   |   |   |-- ✅ Oral
|   |   |       |   |   |   |-- ✅ Phnom_Penh
|   |   |       |   |   |   |-- ✅ Pontianak
|   |   |       |   |   |   |-- ✅ Pyongyang
|   |   |       |   |   |   |-- ✅ Qatar
|   |   |       |   |   |   |-- ✅ Qostanay
|   |   |       |   |   |   |-- ✅ Qyzylorda
|   |   |       |   |   |   |-- ✅ Rangoon
|   |   |       |   |   |   |-- ✅ Riyadh
|   |   |       |   |   |   |-- ✅ Saigon
|   |   |       |   |   |   |-- ✅ Sakhalin
|   |   |       |   |   |   |-- ✅ Samarkand
|   |   |       |   |   |   |-- ✅ Seoul
|   |   |       |   |   |   |-- ✅ Shanghai
|   |   |       |   |   |   |-- ✅ Singapore
|   |   |       |   |   |   |-- ✅ Srednekolymsk
|   |   |       |   |   |   |-- ✅ Taipei
|   |   |       |   |   |   |-- ✅ Tashkent
|   |   |       |   |   |   |-- ✅ Tbilisi
|   |   |       |   |   |   |-- ✅ Tehran
|   |   |       |   |   |   |-- ✅ Tel_Aviv
|   |   |       |   |   |   |-- ✅ Thimbu
|   |   |       |   |   |   |-- ✅ Thimphu
|   |   |       |   |   |   |-- ✅ Tokyo
|   |   |       |   |   |   |-- ✅ Tomsk
|   |   |       |   |   |   |-- ✅ Ujung_Pandang
|   |   |       |   |   |   |-- ✅ Ulaanbaatar
|   |   |       |   |   |   |-- ✅ Ulan_Bator
|   |   |       |   |   |   |-- ✅ Urumqi
|   |   |       |   |   |   |-- ✅ Ust-Nera
|   |   |       |   |   |   |-- ✅ Vientiane
|   |   |       |   |   |   |-- ✅ Vladivostok
|   |   |       |   |   |   |-- ✅ Yakutsk
|   |   |       |   |   |   |-- ✅ Yangon
|   |   |       |   |   |   |-- ✅ Yekaterinburg
|   |   |       |   |   |   \-- ✅ Yerevan
|   |   |       |   |   |-- ✅ Atlantic/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Azores
|   |   |       |   |   |   |-- ✅ Bermuda
|   |   |       |   |   |   |-- ✅ Canary
|   |   |       |   |   |   |-- ✅ Cape_Verde
|   |   |       |   |   |   |-- ✅ Faeroe
|   |   |       |   |   |   |-- ✅ Faroe
|   |   |       |   |   |   |-- ✅ Jan_Mayen
|   |   |       |   |   |   |-- ✅ Madeira
|   |   |       |   |   |   |-- ✅ Reykjavik
|   |   |       |   |   |   |-- ✅ South_Georgia
|   |   |       |   |   |   |-- ✅ St_Helena
|   |   |       |   |   |   \-- ✅ Stanley
|   |   |       |   |   |-- ✅ Australia/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ ACT
|   |   |       |   |   |   |-- ✅ Adelaide
|   |   |       |   |   |   |-- ✅ Brisbane
|   |   |       |   |   |   |-- ✅ Broken_Hill
|   |   |       |   |   |   |-- ✅ Canberra
|   |   |       |   |   |   |-- ✅ Currie
|   |   |       |   |   |   |-- ✅ Darwin
|   |   |       |   |   |   |-- ✅ Eucla
|   |   |       |   |   |   |-- ✅ Hobart
|   |   |       |   |   |   |-- ✅ LHI
|   |   |       |   |   |   |-- ✅ Lindeman
|   |   |       |   |   |   |-- ✅ Lord_Howe
|   |   |       |   |   |   |-- ✅ Melbourne
|   |   |       |   |   |   |-- ✅ North
|   |   |       |   |   |   |-- ✅ NSW
|   |   |       |   |   |   |-- ✅ Perth
|   |   |       |   |   |   |-- ✅ Queensland
|   |   |       |   |   |   |-- ✅ South
|   |   |       |   |   |   |-- ✅ Sydney
|   |   |       |   |   |   |-- ✅ Tasmania
|   |   |       |   |   |   |-- ✅ Victoria
|   |   |       |   |   |   |-- ✅ West
|   |   |       |   |   |   \-- ✅ Yancowinna
|   |   |       |   |   |-- ✅ Brazil/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Acre
|   |   |       |   |   |   |-- ✅ DeNoronha
|   |   |       |   |   |   |-- ✅ East
|   |   |       |   |   |   \-- ✅ West
|   |   |       |   |   |-- ✅ Canada/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Atlantic
|   |   |       |   |   |   |-- ✅ Central
|   |   |       |   |   |   |-- ✅ Eastern
|   |   |       |   |   |   |-- ✅ Mountain
|   |   |       |   |   |   |-- ✅ Newfoundland
|   |   |       |   |   |   |-- ✅ Pacific
|   |   |       |   |   |   |-- ✅ Saskatchewan
|   |   |       |   |   |   \-- ✅ Yukon
|   |   |       |   |   |-- ✅ Chile/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Continental
|   |   |       |   |   |   \-- ✅ EasterIsland
|   |   |       |   |   |-- ✅ Etc/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ GMT
|   |   |       |   |   |   |-- ✅ GMT+0
|   |   |       |   |   |   |-- ✅ GMT+1
|   |   |       |   |   |   |-- ✅ GMT+10
|   |   |       |   |   |   |-- ✅ GMT+11
|   |   |       |   |   |   |-- ✅ GMT+12
|   |   |       |   |   |   |-- ✅ GMT+2
|   |   |       |   |   |   |-- ✅ GMT+3
|   |   |       |   |   |   |-- ✅ GMT+4
|   |   |       |   |   |   |-- ✅ GMT+5
|   |   |       |   |   |   |-- ✅ GMT+6
|   |   |       |   |   |   |-- ✅ GMT+7
|   |   |       |   |   |   |-- ✅ GMT+8
|   |   |       |   |   |   |-- ✅ GMT+9
|   |   |       |   |   |   |-- ✅ GMT-0
|   |   |       |   |   |   |-- ✅ GMT-1
|   |   |       |   |   |   |-- ✅ GMT-10
|   |   |       |   |   |   |-- ✅ GMT-11
|   |   |       |   |   |   |-- ✅ GMT-12
|   |   |       |   |   |   |-- ✅ GMT-13
|   |   |       |   |   |   |-- ✅ GMT-14
|   |   |       |   |   |   |-- ✅ GMT-2
|   |   |       |   |   |   |-- ✅ GMT-3
|   |   |       |   |   |   |-- ✅ GMT-4
|   |   |       |   |   |   |-- ✅ GMT-5
|   |   |       |   |   |   |-- ✅ GMT-6
|   |   |       |   |   |   |-- ✅ GMT-7
|   |   |       |   |   |   |-- ✅ GMT-8
|   |   |       |   |   |   |-- ✅ GMT-9
|   |   |       |   |   |   |-- ✅ GMT0
|   |   |       |   |   |   |-- ✅ Greenwich
|   |   |       |   |   |   |-- ✅ UCT
|   |   |       |   |   |   |-- ✅ Universal
|   |   |       |   |   |   |-- ✅ UTC
|   |   |       |   |   |   \-- ✅ Zulu
|   |   |       |   |   |-- ✅ Europe/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Amsterdam
|   |   |       |   |   |   |-- ✅ Andorra
|   |   |       |   |   |   |-- ✅ Astrakhan
|   |   |       |   |   |   |-- ✅ Athens
|   |   |       |   |   |   |-- ✅ Belfast
|   |   |       |   |   |   |-- ✅ Belgrade
|   |   |       |   |   |   |-- ✅ Berlin
|   |   |       |   |   |   |-- ✅ Bratislava
|   |   |       |   |   |   |-- ✅ Brussels
|   |   |       |   |   |   |-- ✅ Bucharest
|   |   |       |   |   |   |-- ✅ Budapest
|   |   |       |   |   |   |-- ✅ Busingen
|   |   |       |   |   |   |-- ✅ Chisinau
|   |   |       |   |   |   |-- ✅ Copenhagen
|   |   |       |   |   |   |-- ✅ Dublin
|   |   |       |   |   |   |-- ✅ Gibraltar
|   |   |       |   |   |   |-- ✅ Guernsey
|   |   |       |   |   |   |-- ✅ Helsinki
|   |   |       |   |   |   |-- ✅ Isle_of_Man
|   |   |       |   |   |   |-- ✅ Istanbul
|   |   |       |   |   |   |-- ✅ Jersey
|   |   |       |   |   |   |-- ✅ Kaliningrad
|   |   |       |   |   |   |-- ✅ Kiev
|   |   |       |   |   |   |-- ✅ Kirov
|   |   |       |   |   |   |-- ✅ Kyiv
|   |   |       |   |   |   |-- ✅ Lisbon
|   |   |       |   |   |   |-- ✅ Ljubljana
|   |   |       |   |   |   |-- ✅ London
|   |   |       |   |   |   |-- ✅ Luxembourg
|   |   |       |   |   |   |-- ✅ Madrid
|   |   |       |   |   |   |-- ✅ Malta
|   |   |       |   |   |   |-- ✅ Mariehamn
|   |   |       |   |   |   |-- ✅ Minsk
|   |   |       |   |   |   |-- ✅ Monaco
|   |   |       |   |   |   |-- ✅ Moscow
|   |   |       |   |   |   |-- ✅ Nicosia
|   |   |       |   |   |   |-- ✅ Oslo
|   |   |       |   |   |   |-- ✅ Paris
|   |   |       |   |   |   |-- ✅ Podgorica
|   |   |       |   |   |   |-- ✅ Prague
|   |   |       |   |   |   |-- ✅ Riga
|   |   |       |   |   |   |-- ✅ Rome
|   |   |       |   |   |   |-- ✅ Samara
|   |   |       |   |   |   |-- ✅ San_Marino
|   |   |       |   |   |   |-- ✅ Sarajevo
|   |   |       |   |   |   |-- ✅ Saratov
|   |   |       |   |   |   |-- ✅ Simferopol
|   |   |       |   |   |   |-- ✅ Skopje
|   |   |       |   |   |   |-- ✅ Sofia
|   |   |       |   |   |   |-- ✅ Stockholm
|   |   |       |   |   |   |-- ✅ Tallinn
|   |   |       |   |   |   |-- ✅ Tirane
|   |   |       |   |   |   |-- ✅ Tiraspol
|   |   |       |   |   |   |-- ✅ Ulyanovsk
|   |   |       |   |   |   |-- ✅ Uzhgorod
|   |   |       |   |   |   |-- ✅ Vaduz
|   |   |       |   |   |   |-- ✅ Vatican
|   |   |       |   |   |   |-- ✅ Vienna
|   |   |       |   |   |   |-- ✅ Vilnius
|   |   |       |   |   |   |-- ✅ Volgograd
|   |   |       |   |   |   |-- ✅ Warsaw
|   |   |       |   |   |   |-- ✅ Zagreb
|   |   |       |   |   |   |-- ✅ Zaporozhye
|   |   |       |   |   |   \-- ✅ Zurich
|   |   |       |   |   |-- ✅ Indian/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Antananarivo
|   |   |       |   |   |   |-- ✅ Chagos
|   |   |       |   |   |   |-- ✅ Christmas
|   |   |       |   |   |   |-- ✅ Cocos
|   |   |       |   |   |   |-- ✅ Comoro
|   |   |       |   |   |   |-- ✅ Kerguelen
|   |   |       |   |   |   |-- ✅ Mahe
|   |   |       |   |   |   |-- ✅ Maldives
|   |   |       |   |   |   |-- ✅ Mauritius
|   |   |       |   |   |   |-- ✅ Mayotte
|   |   |       |   |   |   \-- ✅ Reunion
|   |   |       |   |   |-- ✅ Mexico/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ BajaNorte
|   |   |       |   |   |   |-- ✅ BajaSur
|   |   |       |   |   |   \-- ✅ General
|   |   |       |   |   |-- ✅ Pacific/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Apia
|   |   |       |   |   |   |-- ✅ Auckland
|   |   |       |   |   |   |-- ✅ Bougainville
|   |   |       |   |   |   |-- ✅ Chatham
|   |   |       |   |   |   |-- ✅ Chuuk
|   |   |       |   |   |   |-- ✅ Easter
|   |   |       |   |   |   |-- ✅ Efate
|   |   |       |   |   |   |-- ✅ Enderbury
|   |   |       |   |   |   |-- ✅ Fakaofo
|   |   |       |   |   |   |-- ✅ Fiji
|   |   |       |   |   |   |-- ✅ Funafuti
|   |   |       |   |   |   |-- ✅ Galapagos
|   |   |       |   |   |   |-- ✅ Gambier
|   |   |       |   |   |   |-- ✅ Guadalcanal
|   |   |       |   |   |   |-- ✅ Guam
|   |   |       |   |   |   |-- ✅ Honolulu
|   |   |       |   |   |   |-- ✅ Johnston
|   |   |       |   |   |   |-- ✅ Kanton
|   |   |       |   |   |   |-- ✅ Kiritimati
|   |   |       |   |   |   |-- ✅ Kosrae
|   |   |       |   |   |   |-- ✅ Kwajalein
|   |   |       |   |   |   |-- ✅ Majuro
|   |   |       |   |   |   |-- ✅ Marquesas
|   |   |       |   |   |   |-- ✅ Midway
|   |   |       |   |   |   |-- ✅ Nauru
|   |   |       |   |   |   |-- ✅ Niue
|   |   |       |   |   |   |-- ✅ Norfolk
|   |   |       |   |   |   |-- ✅ Noumea
|   |   |       |   |   |   |-- ✅ Pago_Pago
|   |   |       |   |   |   |-- ✅ Palau
|   |   |       |   |   |   |-- ✅ Pitcairn
|   |   |       |   |   |   |-- ✅ Pohnpei
|   |   |       |   |   |   |-- ✅ Ponape
|   |   |       |   |   |   |-- ✅ Port_Moresby
|   |   |       |   |   |   |-- ✅ Rarotonga
|   |   |       |   |   |   |-- ✅ Saipan
|   |   |       |   |   |   |-- ✅ Samoa
|   |   |       |   |   |   |-- ✅ Tahiti
|   |   |       |   |   |   |-- ✅ Tarawa
|   |   |       |   |   |   |-- ✅ Tongatapu
|   |   |       |   |   |   |-- ✅ Truk
|   |   |       |   |   |   |-- ✅ Wake
|   |   |       |   |   |   |-- ✅ Wallis
|   |   |       |   |   |   \-- ✅ Yap
|   |   |       |   |   |-- ✅ US/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ Alaska
|   |   |       |   |   |   |-- ✅ Aleutian
|   |   |       |   |   |   |-- ✅ Arizona
|   |   |       |   |   |   |-- ✅ Central
|   |   |       |   |   |   |-- ✅ East-Indiana
|   |   |       |   |   |   |-- ✅ Eastern
|   |   |       |   |   |   |-- ✅ Hawaii
|   |   |       |   |   |   |-- ✅ Indiana-Starke
|   |   |       |   |   |   |-- ✅ Michigan
|   |   |       |   |   |   |-- ✅ Mountain
|   |   |       |   |   |   |-- ✅ Pacific
|   |   |       |   |   |   \-- ✅ Samoa
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ CET
|   |   |       |   |   |-- ✅ CST6CDT
|   |   |       |   |   |-- ✅ Cuba
|   |   |       |   |   |-- ✅ EET
|   |   |       |   |   |-- ✅ Egypt
|   |   |       |   |   |-- ✅ Eire
|   |   |       |   |   |-- ✅ EST
|   |   |       |   |   |-- ✅ EST5EDT
|   |   |       |   |   |-- ✅ Factory
|   |   |       |   |   |-- ✅ GB
|   |   |       |   |   |-- ✅ GB-Eire
|   |   |       |   |   |-- ✅ GMT
|   |   |       |   |   |-- ✅ GMT+0
|   |   |       |   |   |-- ✅ GMT-0
|   |   |       |   |   |-- ✅ GMT0
|   |   |       |   |   |-- ✅ Greenwich
|   |   |       |   |   |-- ✅ Hongkong
|   |   |       |   |   |-- ✅ HST
|   |   |       |   |   |-- ✅ Iceland
|   |   |       |   |   |-- ✅ Iran
|   |   |       |   |   |-- ✅ iso3166.tab
|   |   |       |   |   |-- ✅ Israel
|   |   |       |   |   |-- ✅ Jamaica
|   |   |       |   |   |-- ✅ Japan
|   |   |       |   |   |-- ✅ Kwajalein
|   |   |       |   |   |-- ✅ leapseconds
|   |   |       |   |   |-- ✅ Libya
|   |   |       |   |   |-- ✅ MET
|   |   |       |   |   |-- ✅ MST
|   |   |       |   |   |-- ✅ MST7MDT
|   |   |       |   |   |-- ✅ Navajo
|   |   |       |   |   |-- ✅ NZ
|   |   |       |   |   |-- ✅ NZ-CHAT
|   |   |       |   |   |-- ✅ Poland
|   |   |       |   |   |-- ✅ Portugal
|   |   |       |   |   |-- ✅ PRC
|   |   |       |   |   |-- ✅ PST8PDT
|   |   |       |   |   |-- ✅ ROC
|   |   |       |   |   |-- ✅ ROK
|   |   |       |   |   |-- ✅ Singapore
|   |   |       |   |   |-- ✅ Turkey
|   |   |       |   |   |-- ✅ tzdata.zi
|   |   |       |   |   |-- ✅ UCT
|   |   |       |   |   |-- ✅ Universal
|   |   |       |   |   |-- ✅ UTC
|   |   |       |   |   |-- ✅ W-SU
|   |   |       |   |   |-- ✅ WET
|   |   |       |   |   |-- ✅ zone.tab
|   |   |       |   |   |-- ✅ zone1970.tab
|   |   |       |   |   |-- ✅ zonenow.tab
|   |   |       |   |   \-- ✅ Zulu
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   \-- ✅ zones
|   |   |       |-- ✅ tzdata-2025.2.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ licenses/
|   |   |       |   |   |   \-- ✅ LICENSE_APACHE
|   |   |       |   |   \-- ✅ LICENSE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ tzlocal/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ unix.py
|   |   |       |   |-- ✅ utils.py
|   |   |       |   |-- ✅ win32.py
|   |   |       |   \-- ✅ windows_tz.py
|   |   |       |-- ✅ tzlocal-5.3.1.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ urllib3/
|   |   |       |   |-- ✅ contrib/
|   |   |       |   |   |-- ✅ emscripten/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ connection.py
|   |   |       |   |   |   |-- ✅ emscripten_fetch_worker.js
|   |   |       |   |   |   |-- ✅ fetch.py
|   |   |       |   |   |   |-- ✅ request.py
|   |   |       |   |   |   \-- ✅ response.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ pyopenssl.py
|   |   |       |   |   \-- ✅ socks.py
|   |   |       |   |-- ✅ http2/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ connection.py
|   |   |       |   |   \-- ✅ probe.py
|   |   |       |   |-- ✅ util/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ connection.py
|   |   |       |   |   |-- ✅ proxy.py
|   |   |       |   |   |-- ✅ request.py
|   |   |       |   |   |-- ✅ response.py
|   |   |       |   |   |-- ✅ retry.py
|   |   |       |   |   |-- ✅ ssl_.py
|   |   |       |   |   |-- ✅ ssl_match_hostname.py
|   |   |       |   |   |-- ✅ ssltransport.py
|   |   |       |   |   |-- ✅ timeout.py
|   |   |       |   |   |-- ✅ url.py
|   |   |       |   |   |-- ✅ util.py
|   |   |       |   |   \-- ✅ wait.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _base_connection.py
|   |   |       |   |-- ✅ _collections.py
|   |   |       |   |-- ✅ _request_methods.py
|   |   |       |   |-- ✅ _version.py
|   |   |       |   |-- ✅ connection.py
|   |   |       |   |-- ✅ connectionpool.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ fields.py
|   |   |       |   |-- ✅ filepost.py
|   |   |       |   |-- ✅ poolmanager.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   \-- ✅ response.py
|   |   |       |-- ✅ urllib3-2.5.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ uvicorn/
|   |   |       |   |-- ✅ lifespan/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ off.py
|   |   |       |   |   \-- ✅ on.py
|   |   |       |   |-- ✅ loops/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ asyncio.py
|   |   |       |   |   |-- ✅ auto.py
|   |   |       |   |   \-- ✅ uvloop.py
|   |   |       |   |-- ✅ middleware/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ asgi2.py
|   |   |       |   |   |-- ✅ message_logger.py
|   |   |       |   |   |-- ✅ proxy_headers.py
|   |   |       |   |   \-- ✅ wsgi.py
|   |   |       |   |-- ✅ protocols/
|   |   |       |   |   |-- ✅ http/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ auto.py
|   |   |       |   |   |   |-- ✅ flow_control.py
|   |   |       |   |   |   |-- ✅ h11_impl.py
|   |   |       |   |   |   \-- ✅ httptools_impl.py
|   |   |       |   |   |-- ✅ websockets/
|   |   |       |   |   |   |-- ✅ __init__.py
|   |   |       |   |   |   |-- ✅ auto.py
|   |   |       |   |   |   |-- ✅ websockets_impl.py
|   |   |       |   |   |   |-- ✅ websockets_sansio_impl.py
|   |   |       |   |   |   \-- ✅ wsproto_impl.py
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   \-- ✅ utils.py
|   |   |       |   |-- ✅ supervisors/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ basereload.py
|   |   |       |   |   |-- ✅ multiprocess.py
|   |   |       |   |   |-- ✅ statreload.py
|   |   |       |   |   \-- ✅ watchfilesreload.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __main__.py
|   |   |       |   |-- ✅ _compat.py
|   |   |       |   |-- ✅ _subprocess.py
|   |   |       |   |-- ✅ _types.py
|   |   |       |   |-- ✅ config.py
|   |   |       |   |-- ✅ importer.py
|   |   |       |   |-- ✅ logging.py
|   |   |       |   |-- ✅ main.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ server.py
|   |   |       |   \-- ✅ workers.py
|   |   |       |-- ✅ uvicorn-0.37.0.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   \-- ✅ LICENSE.md
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ websockets/
|   |   |       |   |-- ✅ asyncio/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ async_timeout.py
|   |   |       |   |   |-- ✅ client.py
|   |   |       |   |   |-- ✅ compatibility.py
|   |   |       |   |   |-- ✅ connection.py
|   |   |       |   |   |-- ✅ messages.py
|   |   |       |   |   |-- ✅ router.py
|   |   |       |   |   \-- ✅ server.py
|   |   |       |   |-- ✅ extensions/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ base.py
|   |   |       |   |   \-- ✅ permessage_deflate.py
|   |   |       |   |-- ✅ legacy/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ auth.py
|   |   |       |   |   |-- ✅ client.py
|   |   |       |   |   |-- ✅ exceptions.py
|   |   |       |   |   |-- ✅ framing.py
|   |   |       |   |   |-- ✅ handshake.py
|   |   |       |   |   |-- ✅ http.py
|   |   |       |   |   |-- ✅ protocol.py
|   |   |       |   |   \-- ✅ server.py
|   |   |       |   |-- ✅ sync/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ client.py
|   |   |       |   |   |-- ✅ connection.py
|   |   |       |   |   |-- ✅ messages.py
|   |   |       |   |   |-- ✅ router.py
|   |   |       |   |   |-- ✅ server.py
|   |   |       |   |   \-- ✅ utils.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ __main__.py
|   |   |       |   |-- ✅ auth.py
|   |   |       |   |-- ✅ cli.py
|   |   |       |   |-- ✅ client.py
|   |   |       |   |-- ✅ connection.py
|   |   |       |   |-- ✅ datastructures.py
|   |   |       |   |-- ✅ exceptions.py
|   |   |       |   |-- ✅ frames.py
|   |   |       |   |-- ✅ headers.py
|   |   |       |   |-- ✅ http.py
|   |   |       |   |-- ✅ http11.py
|   |   |       |   |-- ✅ imports.py
|   |   |       |   |-- ✅ protocol.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ server.py
|   |   |       |   |-- ✅ speedups.c
|   |   |       |   |-- ✅ speedups.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ speedups.pyi
|   |   |       |   |-- ✅ streams.py
|   |   |       |   |-- ✅ typing.py
|   |   |       |   |-- ✅ uri.py
|   |   |       |   |-- ✅ utils.py
|   |   |       |   \-- ✅ version.py
|   |   |       |-- ✅ websockets-15.0.1.dist-info/
|   |   |       |   |-- ✅ entry_points.txt
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ LICENSE
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ xgboost/
|   |   |       |   |-- ✅ dask/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ data.py
|   |   |       |   |   \-- ✅ utils.py
|   |   |       |   |-- ✅ lib/
|   |   |       |   |   \-- ✅ xgboost.dll
|   |   |       |   |-- ✅ spark/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ core.py
|   |   |       |   |   |-- ✅ data.py
|   |   |       |   |   |-- ✅ estimator.py
|   |   |       |   |   |-- ✅ params.py
|   |   |       |   |   |-- ✅ summary.py
|   |   |       |   |   \-- ✅ utils.py
|   |   |       |   |-- ✅ testing/
|   |   |       |   |   |-- ✅ __init__.py
|   |   |       |   |   |-- ✅ continuation.py
|   |   |       |   |   |-- ✅ dask.py
|   |   |       |   |   |-- ✅ data.py
|   |   |       |   |   |-- ✅ data_iter.py
|   |   |       |   |   |-- ✅ federated.py
|   |   |       |   |   |-- ✅ metrics.py
|   |   |       |   |   |-- ✅ params.py
|   |   |       |   |   |-- ✅ quantile_dmatrix.py
|   |   |       |   |   |-- ✅ ranking.py
|   |   |       |   |   |-- ✅ shared.py
|   |   |       |   |   \-- ✅ updater.py
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _data_utils.py
|   |   |       |   |-- ✅ _typing.py
|   |   |       |   |-- ✅ callback.py
|   |   |       |   |-- ✅ collective.py
|   |   |       |   |-- ✅ compat.py
|   |   |       |   |-- ✅ config.py
|   |   |       |   |-- ✅ core.py
|   |   |       |   |-- ✅ data.py
|   |   |       |   |-- ✅ federated.py
|   |   |       |   |-- ✅ libpath.py
|   |   |       |   |-- ✅ plotting.py
|   |   |       |   |-- ✅ py.typed
|   |   |       |   |-- ✅ sklearn.py
|   |   |       |   |-- ✅ tracker.py
|   |   |       |   |-- ✅ training.py
|   |   |       |   \-- ✅ VERSION
|   |   |       |-- ✅ xgboost-3.0.5.dist-info/
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ REQUESTED
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ yarl/
|   |   |       |   |-- ✅ __init__.py
|   |   |       |   |-- ✅ _parse.py
|   |   |       |   |-- ✅ _path.py
|   |   |       |   |-- ✅ _query.py
|   |   |       |   |-- ✅ _quoters.py
|   |   |       |   |-- ✅ _quoting.py
|   |   |       |   |-- ✅ _quoting_c.cp312-win_amd64.pyd
|   |   |       |   |-- ✅ _quoting_c.pyx
|   |   |       |   |-- ✅ _quoting_py.py
|   |   |       |   |-- ✅ _url.py
|   |   |       |   \-- ✅ py.typed
|   |   |       |-- ✅ yarl-1.20.1.dist-info/
|   |   |       |   |-- ✅ licenses/
|   |   |       |   |   |-- ✅ LICENSE
|   |   |       |   |   \-- ✅ NOTICE
|   |   |       |   |-- ✅ INSTALLER
|   |   |       |   |-- ✅ METADATA
|   |   |       |   |-- ✅ RECORD
|   |   |       |   |-- ✅ top_level.txt
|   |   |       |   \-- ✅ WHEEL
|   |   |       |-- ✅ _cffi_backend.cp312-win_amd64.pyd
|   |   |       |-- ✅ pylab.py
|   |   |       |-- ✅ scipy-1.16.2-cp312-cp312-win_amd64.whl
|   |   |       |-- ✅ six.py
|   |   |       |-- ✅ threadpoolctl.py
|   |   |       \-- ✅ typing_extensions.py
|   |   |-- ✅ Scripts/
|   |   |   |-- ✅ activate
|   |   |   |-- ✅ activate.bat
|   |   |   |-- ✅ Activate.ps1
|   |   |   |-- ✅ dateparser-download.exe
|   |   |   |-- ✅ deactivate.bat
|   |   |   |-- ✅ dotenv.exe
|   |   |   |-- ✅ f2py.exe
|   |   |   |-- ✅ fastapi.exe
|   |   |   |-- ✅ fonttools.exe
|   |   |   |-- ✅ httpx.exe
|   |   |   |-- ✅ normalizer.exe
|   |   |   |-- ✅ numpy-config.exe
|   |   |   |-- ✅ pip.exe
|   |   |   |-- ✅ pip3.12.exe
|   |   |   |-- ✅ pip3.exe
|   |   |   |-- ✅ pyftmerge.exe
|   |   |   |-- ✅ pyftsubset.exe
|   |   |   |-- ✅ python.exe
|   |   |   |-- ✅ pythonw.exe
|   |   |   |-- ✅ ttx.exe
|   |   |   |-- ✅ uvicorn.exe
|   |   |   \-- ✅ websockets.exe
|   |   |-- ✅ share/
|   |   |   \-- ✅ man/
|   |   |       \-- ✅ man1/
|   |   |           \-- ✅ ttx.1
|   |   \-- ✅ pyvenv.cfg
|   |-- ✅ data/
|   |   \-- ✅ trades.db
|   |-- ✅ routes/
|   |   |-- ✅ __init__.py
|   |   |-- ✅ ai.py
|   |   |-- ✅ backtest.py
|   |   |-- ✅ binance.py
|   |   |-- ✅ candles.py
|   |   |-- ✅ chart.py
|   |   |-- ✅ external_data.py
|   |   |-- ✅ health.py
|   |   |-- ✅ prices.py
|   |   |-- ✅ settings.py
|   |   |-- ✅ signals.py
|   |   |-- ✅ stats.py
|   |   |-- ✅ stress.py
|   |   |-- ✅ trade_logs.py
|   |   |-- ✅ trades.py
|   |   \-- ✅ ws.py
|   |-- ✅ scripts/
|   |   |-- ✅ adapter_smoke.py
|   |   \-- ✅ check_dev_deps_in_runtime.py
|   |-- ✅ testing/
|   |   |-- ✅ __init__.py
|   |   \-- ✅ mock_signals.py
|   |-- ✅ tests/
|   |   |-- ✅ utils/
|   |   |   |-- ✅ __init__.py
|   |   |   \-- ✅ mock_signals.py
|   |   |-- ✅ __init__.py
|   |   |-- ✅ conftest.py
|   |   |-- ✅ test_adapters.py
|   |   |-- ✅ test_ai_pipeline.py
|   |   |-- ✅ test_api.py
|   |   |-- ✅ test_binance.py
|   |   |-- ✅ test_chart.py
|   |   |-- ✅ test_demo_endpoints.py
|   |   |-- ✅ test_exchanges.py
|   |   |-- ✅ test_exchanges_adapter_alias.py
|   |   |-- ✅ test_exchanges_factory.py
|   |   |-- ✅ test_market_data.py
|   |   |-- ✅ test_risk.py
|   |   |-- ✅ test_settings.py
|   |   |-- ✅ test_signals_api.py
|   |   |-- ✅ test_stats.py
|   |   |-- ✅ test_stress_route.py
|   |   |-- ✅ test_trade_logger.py
|   |   |-- ✅ test_trades.py
|   |   |-- ✅ test_trades_integration.py
|   |   \-- ✅ test_train_and_save.py
|   |-- ✅ utils/
|   |   |-- ✅ __init__.py
|   |   |-- ✅ analytics.py
|   |   |-- ✅ analytics.py.bak
|   |   |-- ✅ binance_client.py
|   |   |-- ✅ cryptopanic_client.py
|   |   |-- ✅ db.py
|   |   |-- ✅ exchanges.py
|   |   |-- ✅ failsafe.py
|   |   |-- ✅ logging.py
|   |   |-- ✅ market_data.py
|   |   |-- ✅ metrics.py
|   |   |-- ✅ pnl.py
|   |   |-- ✅ pnl.py.bak
|   |   |-- ✅ risk.py
|   |   |-- ✅ seed_candles.py
|   |   |-- ✅ seed_candles.py.bak
|   |   |-- ✅ startup.py
|   |   |-- ✅ trade_logger.py
|   |   |-- ✅ trade_logger.pyi
|   |   |-- ✅ trade_utils.py
|   |   \-- ✅ twitter_client.py
|   |-- ✅ .env.example
|   |-- ✅ __init__.py
|   |-- ✅ config.py
|   |-- ✅ database.py
|   |-- ✅ database.pyi
|   |-- ✅ Dockerfile
|   |-- ✅ main.py
|   |-- ✅ Makefile
|   |-- ✅ README-ccxt.md
|   |-- ✅ README.md
|   |-- ✅ requirements-ccxt.txt
|   |-- ✅ requirements-dev.txt
|   |-- ✅ requirements-optional.txt
|   |-- ✅ requirements.txt
|   |-- ✅ run_debug.py
|   \-- ✅ seed_trades.py
|-- ✅ config/
|   |-- ✅ stress/
|   |   \-- ✅ experiments.json
|   |-- ✅ .env.example
|   |-- ✅ __init__.py
|   |-- ✅ config.py
|   |-- ✅ config.pyi
|   \-- ✅ README.md
|-- ✅ database/
|   |-- ✅ analytics.py
|   |-- ✅ init_db.py
|   \-- ✅ schema.sql
|-- ✅ docs/
|   |-- ✅ architecture.puml
|   |-- ✅ architecture.svg
|   \-- ✅ trade-sequence.mmd
|-- ✅ frontend/
|   |-- ✅ .github/
|   |   \-- ✅ workflows/
|   |       \-- ✅ frontend-ci.yml
|   |-- ✅ backups/
|   |   \-- ✅ legacy/
|   |       \-- ✅ api.js
|   |-- ✅ frontend/
|   |   \-- ✅ backups/
|   |       \-- ✅ legacy/
|   |           |-- ✅ App.restored.tsx
|   |           |-- ✅ src_api_client.jsx.bak
|   |           |-- ✅ src_components_AnalyticsCards.jsx.bak
|   |           |-- ✅ src_components_ApiTest.jsx.bak
|   |           |-- ✅ src_components_BalanceCard.jsx.bak
|   |           |-- ✅ src_components_CandlesChart.jsx.bak
|   |           |-- ✅ src_components_Chart.jsx.bak
|   |           |-- ✅ src_components_ChartView.jsx.bak
|   |           |-- ✅ src_components_EquityChart.jsx.bak
|   |           |-- ✅ src_components_ErrorBanner.jsx.bak
|   |           |-- ✅ src_components_Header.jsx.bak
|   |           |-- ✅ src_components_LoaderOverlay.jsx.bak
|   |           |-- ✅ src_components_RiskCards.jsx.bak
|   |           |-- ✅ src_components_RiskMonitor.jsx.bak
|   |           |-- ✅ src_components_Sidebar.jsx.bak
|   |           |-- ✅ src_components_StatsCard.jsx.bak
|   |           |-- ✅ src_components_Toast.jsx.bak
|   |           |-- ✅ src_components_TradeLogs.jsx.bak
|   |           |-- ✅ src_components_TradeTable.jsx.bak
|   |           |-- ✅ src_components_Watchlist.jsx.bak
|   |           |-- ✅ src_hooks_useAutoRefresh.jsx.bak
|   |           |-- ✅ src_hooks_useDarkMode.jsx.bak
|   |           |-- ✅ src_hooks_useDashboardData.jsx.bak
|   |           |-- ✅ src_index.jsx.bak
|   |           |-- ✅ src_main.jsx.bak
|   |           |-- ✅ src_pages_Backtest.jsx.bak
|   |           |-- ✅ src_pages_Dashboard.jsx.bak
|   |           |-- ✅ src_pages_Settings.jsx.bak
|   |           \-- ✅ src_pages_Trades.jsx.bak
|   |-- ✅ legacy/
|   |   |-- ✅ App.jsx.bak
|   |   |-- ✅ index.js.bak
|   |   |-- ✅ main.jsx.bak
|   |   |-- ✅ PriceChart.jsx.bak
|   |   |-- ✅ SentimentPanel.jsx.bak
|   |   |-- ✅ SignalsList.jsx.bak
|   |   |-- ✅ TradeForm.jsx.bak
|   |   |-- ✅ TradeHistory.jsx.bak
|   |   \-- ✅ TradingForm.jsx.bak
|   |-- ✅ node_modules/
|   |   |-- ✅ .bin/
|   |   |   |-- ✅ autoprefixer
|   |   |   |-- ✅ autoprefixer.cmd
|   |   |   |-- ✅ autoprefixer.ps1
|   |   |   |-- ✅ baseline-browser-mapping
|   |   |   |-- ✅ baseline-browser-mapping.cmd
|   |   |   |-- ✅ baseline-browser-mapping.ps1
|   |   |   |-- ✅ browserslist
|   |   |   |-- ✅ browserslist.cmd
|   |   |   |-- ✅ browserslist.ps1
|   |   |   |-- ✅ cssesc
|   |   |   |-- ✅ cssesc.cmd
|   |   |   |-- ✅ cssesc.ps1
|   |   |   |-- ✅ esbuild
|   |   |   |-- ✅ esbuild.cmd
|   |   |   |-- ✅ esbuild.ps1
|   |   |   |-- ✅ glob
|   |   |   |-- ✅ glob.cmd
|   |   |   |-- ✅ glob.ps1
|   |   |   |-- ✅ jiti
|   |   |   |-- ✅ jiti.cmd
|   |   |   |-- ✅ jiti.ps1
|   |   |   |-- ✅ jsesc
|   |   |   |-- ✅ jsesc.cmd
|   |   |   |-- ✅ jsesc.ps1
|   |   |   |-- ✅ json5
|   |   |   |-- ✅ json5.cmd
|   |   |   |-- ✅ json5.ps1
|   |   |   |-- ✅ loose-envify
|   |   |   |-- ✅ loose-envify.cmd
|   |   |   |-- ✅ loose-envify.ps1
|   |   |   |-- ✅ lz-string
|   |   |   |-- ✅ lz-string.cmd
|   |   |   |-- ✅ lz-string.ps1
|   |   |   |-- ✅ nanoid
|   |   |   |-- ✅ nanoid.cmd
|   |   |   |-- ✅ nanoid.ps1
|   |   |   |-- ✅ node-which
|   |   |   |-- ✅ node-which.cmd
|   |   |   |-- ✅ node-which.ps1
|   |   |   |-- ✅ parser
|   |   |   |-- ✅ parser.cmd
|   |   |   |-- ✅ parser.ps1
|   |   |   |-- ✅ resolve
|   |   |   |-- ✅ resolve.cmd
|   |   |   |-- ✅ resolve.ps1
|   |   |   |-- ✅ rollup
|   |   |   |-- ✅ rollup.cmd
|   |   |   |-- ✅ rollup.ps1
|   |   |   |-- ✅ semver
|   |   |   |-- ✅ semver.cmd
|   |   |   |-- ✅ semver.ps1
|   |   |   |-- ✅ sucrase
|   |   |   |-- ✅ sucrase-node
|   |   |   |-- ✅ sucrase-node.cmd
|   |   |   |-- ✅ sucrase-node.ps1
|   |   |   |-- ✅ sucrase.cmd
|   |   |   |-- ✅ sucrase.ps1
|   |   |   |-- ✅ tailwind
|   |   |   |-- ✅ tailwind.cmd
|   |   |   |-- ✅ tailwind.ps1
|   |   |   |-- ✅ tailwindcss
|   |   |   |-- ✅ tailwindcss.cmd
|   |   |   |-- ✅ tailwindcss.ps1
|   |   |   |-- ✅ tsc
|   |   |   |-- ✅ tsc.cmd
|   |   |   |-- ✅ tsc.ps1
|   |   |   |-- ✅ tsserver
|   |   |   |-- ✅ tsserver.cmd
|   |   |   |-- ✅ tsserver.ps1
|   |   |   |-- ✅ update-browserslist-db
|   |   |   |-- ✅ update-browserslist-db.cmd
|   |   |   |-- ✅ update-browserslist-db.ps1
|   |   |   |-- ✅ vite
|   |   |   |-- ✅ vite-node
|   |   |   |-- ✅ vite-node.cmd
|   |   |   |-- ✅ vite-node.ps1
|   |   |   |-- ✅ vite.cmd
|   |   |   |-- ✅ vite.ps1
|   |   |   |-- ✅ vitest
|   |   |   |-- ✅ vitest.cmd
|   |   |   |-- ✅ vitest.ps1
|   |   |   |-- ✅ why-is-node-running
|   |   |   |-- ✅ why-is-node-running.cmd
|   |   |   |-- ✅ why-is-node-running.ps1
|   |   |   |-- ✅ yaml
|   |   |   |-- ✅ yaml.cmd
|   |   |   \-- ✅ yaml.ps1
|   |   |-- ✅ .vite/
|   |   |   \-- ✅ deps/
|   |   |       |-- ✅ _metadata.json
|   |   |       |-- ✅ axios.js
|   |   |       |-- ✅ axios.js.map
|   |   |       |-- ✅ chunk-HKJ2B2AA.js
|   |   |       |-- ✅ chunk-HKJ2B2AA.js.map
|   |   |       |-- ✅ chunk-IU5JCZQL.js
|   |   |       |-- ✅ chunk-IU5JCZQL.js.map
|   |   |       |-- ✅ chunk-TSRML2OT.js
|   |   |       |-- ✅ chunk-TSRML2OT.js.map
|   |   |       |-- ✅ package.json
|   |   |       |-- ✅ react-dom.js
|   |   |       |-- ✅ react-dom.js.map
|   |   |       |-- ✅ react-dom_client.js
|   |   |       |-- ✅ react-dom_client.js.map
|   |   |       |-- ✅ react.js
|   |   |       |-- ✅ react.js.map
|   |   |       |-- ✅ react_jsx-dev-runtime.js
|   |   |       |-- ✅ react_jsx-dev-runtime.js.map
|   |   |       |-- ✅ react_jsx-runtime.js
|   |   |       \-- ✅ react_jsx-runtime.js.map
|   |   |-- ✅ .vite-temp/
|   |   |-- ✅ @adobe/
|   |   |   \-- ✅ css-tools/
|   |   |       |-- ✅ dist/
|   |   |       |   |-- ✅ cjs/
|   |   |       |   |   |-- ✅ adobe-css-tools.cjs
|   |   |       |   |   |-- ✅ adobe-css-tools.cjs.map
|   |   |       |   |   \-- ✅ adobe-css-tools.d.cts
|   |   |       |   |-- ✅ esm/
|   |   |       |   |   |-- ✅ adobe-css-tools.d.mts
|   |   |       |   |   |-- ✅ adobe-css-tools.mjs
|   |   |       |   |   \-- ✅ adobe-css-tools.mjs.map
|   |   |       |   \-- ✅ umd/
|   |   |       |       |-- ✅ adobe-css-tools.d.ts
|   |   |       |       |-- ✅ adobe-css-tools.js
|   |   |       |       \-- ✅ adobe-css-tools.js.map
|   |   |       |-- ✅ docs/
|   |   |       |   |-- ✅ API.md
|   |   |       |   |-- ✅ AST.md
|   |   |       |   |-- ✅ CHANGELOG.md
|   |   |       |   \-- ✅ EXAMPLES.md
|   |   |       |-- ✅ LICENSE
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @alloc/
|   |   |   \-- ✅ quick-lru/
|   |   |       |-- ✅ index.d.ts
|   |   |       |-- ✅ index.js
|   |   |       |-- ✅ license
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ readme.md
|   |   |-- ✅ @babel/
|   |   |   |-- ✅ code-frame/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   \-- ✅ index.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ compat-data/
|   |   |   |   |-- ✅ data/
|   |   |   |   |   |-- ✅ corejs2-built-ins.json
|   |   |   |   |   |-- ✅ corejs3-shipped-proposals.json
|   |   |   |   |   |-- ✅ native-modules.json
|   |   |   |   |   |-- ✅ overlapping-plugins.json
|   |   |   |   |   |-- ✅ plugin-bugfixes.json
|   |   |   |   |   \-- ✅ plugins.json
|   |   |   |   |-- ✅ corejs2-built-ins.js
|   |   |   |   |-- ✅ corejs3-shipped-proposals.js
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ native-modules.js
|   |   |   |   |-- ✅ overlapping-plugins.js
|   |   |   |   |-- ✅ package.json
|   |   |   |   |-- ✅ plugin-bugfixes.js
|   |   |   |   |-- ✅ plugins.js
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ core/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ config/
|   |   |   |   |   |   |-- ✅ files/
|   |   |   |   |   |   |   |-- ✅ configuration.js
|   |   |   |   |   |   |   |-- ✅ configuration.js.map
|   |   |   |   |   |   |   |-- ✅ import.cjs
|   |   |   |   |   |   |   |-- ✅ import.cjs.map
|   |   |   |   |   |   |   |-- ✅ index-browser.js
|   |   |   |   |   |   |   |-- ✅ index-browser.js.map
|   |   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |   |   |-- ✅ module-types.js
|   |   |   |   |   |   |   |-- ✅ module-types.js.map
|   |   |   |   |   |   |   |-- ✅ package.js
|   |   |   |   |   |   |   |-- ✅ package.js.map
|   |   |   |   |   |   |   |-- ✅ plugins.js
|   |   |   |   |   |   |   |-- ✅ plugins.js.map
|   |   |   |   |   |   |   |-- ✅ types.js
|   |   |   |   |   |   |   |-- ✅ types.js.map
|   |   |   |   |   |   |   |-- ✅ utils.js
|   |   |   |   |   |   |   \-- ✅ utils.js.map
|   |   |   |   |   |   |-- ✅ helpers/
|   |   |   |   |   |   |   |-- ✅ config-api.js
|   |   |   |   |   |   |   |-- ✅ config-api.js.map
|   |   |   |   |   |   |   |-- ✅ deep-array.js
|   |   |   |   |   |   |   |-- ✅ deep-array.js.map
|   |   |   |   |   |   |   |-- ✅ environment.js
|   |   |   |   |   |   |   \-- ✅ environment.js.map
|   |   |   |   |   |   |-- ✅ validation/
|   |   |   |   |   |   |   |-- ✅ option-assertions.js
|   |   |   |   |   |   |   |-- ✅ option-assertions.js.map
|   |   |   |   |   |   |   |-- ✅ options.js
|   |   |   |   |   |   |   |-- ✅ options.js.map
|   |   |   |   |   |   |   |-- ✅ plugins.js
|   |   |   |   |   |   |   |-- ✅ plugins.js.map
|   |   |   |   |   |   |   |-- ✅ removed.js
|   |   |   |   |   |   |   \-- ✅ removed.js.map
|   |   |   |   |   |   |-- ✅ cache-contexts.js
|   |   |   |   |   |   |-- ✅ cache-contexts.js.map
|   |   |   |   |   |   |-- ✅ caching.js
|   |   |   |   |   |   |-- ✅ caching.js.map
|   |   |   |   |   |   |-- ✅ config-chain.js
|   |   |   |   |   |   |-- ✅ config-chain.js.map
|   |   |   |   |   |   |-- ✅ config-descriptors.js
|   |   |   |   |   |   |-- ✅ config-descriptors.js.map
|   |   |   |   |   |   |-- ✅ full.js
|   |   |   |   |   |   |-- ✅ full.js.map
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |   |-- ✅ item.js
|   |   |   |   |   |   |-- ✅ item.js.map
|   |   |   |   |   |   |-- ✅ partial.js
|   |   |   |   |   |   |-- ✅ partial.js.map
|   |   |   |   |   |   |-- ✅ pattern-to-regex.js
|   |   |   |   |   |   |-- ✅ pattern-to-regex.js.map
|   |   |   |   |   |   |-- ✅ plugin.js
|   |   |   |   |   |   |-- ✅ plugin.js.map
|   |   |   |   |   |   |-- ✅ printer.js
|   |   |   |   |   |   |-- ✅ printer.js.map
|   |   |   |   |   |   |-- ✅ resolve-targets-browser.js
|   |   |   |   |   |   |-- ✅ resolve-targets-browser.js.map
|   |   |   |   |   |   |-- ✅ resolve-targets.js
|   |   |   |   |   |   |-- ✅ resolve-targets.js.map
|   |   |   |   |   |   |-- ✅ util.js
|   |   |   |   |   |   \-- ✅ util.js.map
|   |   |   |   |   |-- ✅ errors/
|   |   |   |   |   |   |-- ✅ config-error.js
|   |   |   |   |   |   |-- ✅ config-error.js.map
|   |   |   |   |   |   |-- ✅ rewrite-stack-trace.js
|   |   |   |   |   |   \-- ✅ rewrite-stack-trace.js.map
|   |   |   |   |   |-- ✅ gensync-utils/
|   |   |   |   |   |   |-- ✅ async.js
|   |   |   |   |   |   |-- ✅ async.js.map
|   |   |   |   |   |   |-- ✅ fs.js
|   |   |   |   |   |   |-- ✅ fs.js.map
|   |   |   |   |   |   |-- ✅ functional.js
|   |   |   |   |   |   \-- ✅ functional.js.map
|   |   |   |   |   |-- ✅ parser/
|   |   |   |   |   |   |-- ✅ util/
|   |   |   |   |   |   |   |-- ✅ missing-plugin-helper.js
|   |   |   |   |   |   |   \-- ✅ missing-plugin-helper.js.map
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ index.js.map
|   |   |   |   |   |-- ✅ tools/
|   |   |   |   |   |   |-- ✅ build-external-helpers.js
|   |   |   |   |   |   \-- ✅ build-external-helpers.js.map
|   |   |   |   |   |-- ✅ transformation/
|   |   |   |   |   |   |-- ✅ file/
|   |   |   |   |   |   |   |-- ✅ babel-7-helpers.cjs
|   |   |   |   |   |   |   |-- ✅ babel-7-helpers.cjs.map
|   |   |   |   |   |   |   |-- ✅ file.js
|   |   |   |   |   |   |   |-- ✅ file.js.map
|   |   |   |   |   |   |   |-- ✅ generate.js
|   |   |   |   |   |   |   |-- ✅ generate.js.map
|   |   |   |   |   |   |   |-- ✅ merge-map.js
|   |   |   |   |   |   |   \-- ✅ merge-map.js.map
|   |   |   |   |   |   |-- ✅ util/
|   |   |   |   |   |   |   |-- ✅ clone-deep.js
|   |   |   |   |   |   |   \-- ✅ clone-deep.js.map
|   |   |   |   |   |   |-- ✅ block-hoist-plugin.js
|   |   |   |   |   |   |-- ✅ block-hoist-plugin.js.map
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |   |-- ✅ normalize-file.js
|   |   |   |   |   |   |-- ✅ normalize-file.js.map
|   |   |   |   |   |   |-- ✅ normalize-opts.js
|   |   |   |   |   |   |-- ✅ normalize-opts.js.map
|   |   |   |   |   |   |-- ✅ plugin-pass.js
|   |   |   |   |   |   \-- ✅ plugin-pass.js.map
|   |   |   |   |   |-- ✅ vendor/
|   |   |   |   |   |   |-- ✅ import-meta-resolve.js
|   |   |   |   |   |   \-- ✅ import-meta-resolve.js.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ parse.js
|   |   |   |   |   |-- ✅ parse.js.map
|   |   |   |   |   |-- ✅ transform-ast.js
|   |   |   |   |   |-- ✅ transform-ast.js.map
|   |   |   |   |   |-- ✅ transform-file-browser.js
|   |   |   |   |   |-- ✅ transform-file-browser.js.map
|   |   |   |   |   |-- ✅ transform-file.js
|   |   |   |   |   |-- ✅ transform-file.js.map
|   |   |   |   |   |-- ✅ transform.js
|   |   |   |   |   \-- ✅ transform.js.map
|   |   |   |   |-- ✅ src/
|   |   |   |   |   |-- ✅ config/
|   |   |   |   |   |   |-- ✅ files/
|   |   |   |   |   |   |   |-- ✅ index-browser.ts
|   |   |   |   |   |   |   \-- ✅ index.ts
|   |   |   |   |   |   |-- ✅ resolve-targets-browser.ts
|   |   |   |   |   |   \-- ✅ resolve-targets.ts
|   |   |   |   |   |-- ✅ transform-file-browser.ts
|   |   |   |   |   \-- ✅ transform-file.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ generator/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ generators/
|   |   |   |   |   |   |-- ✅ base.js
|   |   |   |   |   |   |-- ✅ base.js.map
|   |   |   |   |   |   |-- ✅ classes.js
|   |   |   |   |   |   |-- ✅ classes.js.map
|   |   |   |   |   |   |-- ✅ deprecated.js
|   |   |   |   |   |   |-- ✅ deprecated.js.map
|   |   |   |   |   |   |-- ✅ expressions.js
|   |   |   |   |   |   |-- ✅ expressions.js.map
|   |   |   |   |   |   |-- ✅ flow.js
|   |   |   |   |   |   |-- ✅ flow.js.map
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |   |-- ✅ jsx.js
|   |   |   |   |   |   |-- ✅ jsx.js.map
|   |   |   |   |   |   |-- ✅ methods.js
|   |   |   |   |   |   |-- ✅ methods.js.map
|   |   |   |   |   |   |-- ✅ modules.js
|   |   |   |   |   |   |-- ✅ modules.js.map
|   |   |   |   |   |   |-- ✅ statements.js
|   |   |   |   |   |   |-- ✅ statements.js.map
|   |   |   |   |   |   |-- ✅ template-literals.js
|   |   |   |   |   |   |-- ✅ template-literals.js.map
|   |   |   |   |   |   |-- ✅ types.js
|   |   |   |   |   |   |-- ✅ types.js.map
|   |   |   |   |   |   |-- ✅ typescript.js
|   |   |   |   |   |   \-- ✅ typescript.js.map
|   |   |   |   |   |-- ✅ node/
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |   |-- ✅ parentheses.js
|   |   |   |   |   |   |-- ✅ parentheses.js.map
|   |   |   |   |   |   |-- ✅ whitespace.js
|   |   |   |   |   |   \-- ✅ whitespace.js.map
|   |   |   |   |   |-- ✅ buffer.js
|   |   |   |   |   |-- ✅ buffer.js.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ printer.js
|   |   |   |   |   |-- ✅ printer.js.map
|   |   |   |   |   |-- ✅ source-map.js
|   |   |   |   |   |-- ✅ source-map.js.map
|   |   |   |   |   |-- ✅ token-map.js
|   |   |   |   |   \-- ✅ token-map.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ helper-compilation-targets/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ debug.js
|   |   |   |   |   |-- ✅ debug.js.map
|   |   |   |   |   |-- ✅ filter-items.js
|   |   |   |   |   |-- ✅ filter-items.js.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ options.js
|   |   |   |   |   |-- ✅ options.js.map
|   |   |   |   |   |-- ✅ pretty.js
|   |   |   |   |   |-- ✅ pretty.js.map
|   |   |   |   |   |-- ✅ targets.js
|   |   |   |   |   |-- ✅ targets.js.map
|   |   |   |   |   |-- ✅ utils.js
|   |   |   |   |   \-- ✅ utils.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ helper-globals/
|   |   |   |   |-- ✅ data/
|   |   |   |   |   |-- ✅ browser-upper.json
|   |   |   |   |   |-- ✅ builtin-lower.json
|   |   |   |   |   \-- ✅ builtin-upper.json
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ helper-module-imports/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ import-builder.js
|   |   |   |   |   |-- ✅ import-builder.js.map
|   |   |   |   |   |-- ✅ import-injector.js
|   |   |   |   |   |-- ✅ import-injector.js.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ is-module.js
|   |   |   |   |   \-- ✅ is-module.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ helper-module-transforms/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ dynamic-import.js
|   |   |   |   |   |-- ✅ dynamic-import.js.map
|   |   |   |   |   |-- ✅ get-module-name.js
|   |   |   |   |   |-- ✅ get-module-name.js.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ lazy-modules.js
|   |   |   |   |   |-- ✅ lazy-modules.js.map
|   |   |   |   |   |-- ✅ normalize-and-load-metadata.js
|   |   |   |   |   |-- ✅ normalize-and-load-metadata.js.map
|   |   |   |   |   |-- ✅ rewrite-live-references.js
|   |   |   |   |   |-- ✅ rewrite-live-references.js.map
|   |   |   |   |   |-- ✅ rewrite-this.js
|   |   |   |   |   \-- ✅ rewrite-this.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ helper-plugin-utils/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   \-- ✅ index.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ helper-string-parser/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   \-- ✅ index.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ helper-validator-identifier/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ identifier.js
|   |   |   |   |   |-- ✅ identifier.js.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ keyword.js
|   |   |   |   |   \-- ✅ keyword.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ helper-validator-option/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ find-suggestion.js
|   |   |   |   |   |-- ✅ find-suggestion.js.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ validator.js
|   |   |   |   |   \-- ✅ validator.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ helpers/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ helpers/
|   |   |   |   |   |   |-- ✅ applyDecoratedDescriptor.js
|   |   |   |   |   |   |-- ✅ applyDecoratedDescriptor.js.map
|   |   |   |   |   |   |-- ✅ applyDecs.js
|   |   |   |   |   |   |-- ✅ applyDecs.js.map
|   |   |   |   |   |   |-- ✅ applyDecs2203.js
|   |   |   |   |   |   |-- ✅ applyDecs2203.js.map
|   |   |   |   |   |   |-- ✅ applyDecs2203R.js
|   |   |   |   |   |   |-- ✅ applyDecs2203R.js.map
|   |   |   |   |   |   |-- ✅ applyDecs2301.js
|   |   |   |   |   |   |-- ✅ applyDecs2301.js.map
|   |   |   |   |   |   |-- ✅ applyDecs2305.js
|   |   |   |   |   |   |-- ✅ applyDecs2305.js.map
|   |   |   |   |   |   |-- ✅ applyDecs2311.js
|   |   |   |   |   |   |-- ✅ applyDecs2311.js.map
|   |   |   |   |   |   |-- ✅ arrayLikeToArray.js
|   |   |   |   |   |   |-- ✅ arrayLikeToArray.js.map
|   |   |   |   |   |   |-- ✅ arrayWithHoles.js
|   |   |   |   |   |   |-- ✅ arrayWithHoles.js.map
|   |   |   |   |   |   |-- ✅ arrayWithoutHoles.js
|   |   |   |   |   |   |-- ✅ arrayWithoutHoles.js.map
|   |   |   |   |   |   |-- ✅ assertClassBrand.js
|   |   |   |   |   |   |-- ✅ assertClassBrand.js.map
|   |   |   |   |   |   |-- ✅ assertThisInitialized.js
|   |   |   |   |   |   |-- ✅ assertThisInitialized.js.map
|   |   |   |   |   |   |-- ✅ asyncGeneratorDelegate.js
|   |   |   |   |   |   |-- ✅ asyncGeneratorDelegate.js.map
|   |   |   |   |   |   |-- ✅ asyncIterator.js
|   |   |   |   |   |   |-- ✅ asyncIterator.js.map
|   |   |   |   |   |   |-- ✅ asyncToGenerator.js
|   |   |   |   |   |   |-- ✅ asyncToGenerator.js.map
|   |   |   |   |   |   |-- ✅ awaitAsyncGenerator.js
|   |   |   |   |   |   |-- ✅ awaitAsyncGenerator.js.map
|   |   |   |   |   |   |-- ✅ AwaitValue.js
|   |   |   |   |   |   |-- ✅ AwaitValue.js.map
|   |   |   |   |   |   |-- ✅ callSuper.js
|   |   |   |   |   |   |-- ✅ callSuper.js.map
|   |   |   |   |   |   |-- ✅ checkInRHS.js
|   |   |   |   |   |   |-- ✅ checkInRHS.js.map
|   |   |   |   |   |   |-- ✅ checkPrivateRedeclaration.js
|   |   |   |   |   |   |-- ✅ checkPrivateRedeclaration.js.map
|   |   |   |   |   |   |-- ✅ classApplyDescriptorDestructureSet.js
|   |   |   |   |   |   |-- ✅ classApplyDescriptorDestructureSet.js.map
|   |   |   |   |   |   |-- ✅ classApplyDescriptorGet.js
|   |   |   |   |   |   |-- ✅ classApplyDescriptorGet.js.map
|   |   |   |   |   |   |-- ✅ classApplyDescriptorSet.js
|   |   |   |   |   |   |-- ✅ classApplyDescriptorSet.js.map
|   |   |   |   |   |   |-- ✅ classCallCheck.js
|   |   |   |   |   |   |-- ✅ classCallCheck.js.map
|   |   |   |   |   |   |-- ✅ classCheckPrivateStaticAccess.js
|   |   |   |   |   |   |-- ✅ classCheckPrivateStaticAccess.js.map
|   |   |   |   |   |   |-- ✅ classCheckPrivateStaticFieldDescriptor.js
|   |   |   |   |   |   |-- ✅ classCheckPrivateStaticFieldDescriptor.js.map
|   |   |   |   |   |   |-- ✅ classExtractFieldDescriptor.js
|   |   |   |   |   |   |-- ✅ classExtractFieldDescriptor.js.map
|   |   |   |   |   |   |-- ✅ classNameTDZError.js
|   |   |   |   |   |   |-- ✅ classNameTDZError.js.map
|   |   |   |   |   |   |-- ✅ classPrivateFieldDestructureSet.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldDestructureSet.js.map
|   |   |   |   |   |   |-- ✅ classPrivateFieldGet.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldGet.js.map
|   |   |   |   |   |   |-- ✅ classPrivateFieldGet2.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldGet2.js.map
|   |   |   |   |   |   |-- ✅ classPrivateFieldInitSpec.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldInitSpec.js.map
|   |   |   |   |   |   |-- ✅ classPrivateFieldLooseBase.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldLooseBase.js.map
|   |   |   |   |   |   |-- ✅ classPrivateFieldLooseKey.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldLooseKey.js.map
|   |   |   |   |   |   |-- ✅ classPrivateFieldSet.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldSet.js.map
|   |   |   |   |   |   |-- ✅ classPrivateFieldSet2.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldSet2.js.map
|   |   |   |   |   |   |-- ✅ classPrivateGetter.js
|   |   |   |   |   |   |-- ✅ classPrivateGetter.js.map
|   |   |   |   |   |   |-- ✅ classPrivateMethodGet.js
|   |   |   |   |   |   |-- ✅ classPrivateMethodGet.js.map
|   |   |   |   |   |   |-- ✅ classPrivateMethodInitSpec.js
|   |   |   |   |   |   |-- ✅ classPrivateMethodInitSpec.js.map
|   |   |   |   |   |   |-- ✅ classPrivateMethodSet.js
|   |   |   |   |   |   |-- ✅ classPrivateMethodSet.js.map
|   |   |   |   |   |   |-- ✅ classPrivateSetter.js
|   |   |   |   |   |   |-- ✅ classPrivateSetter.js.map
|   |   |   |   |   |   |-- ✅ classStaticPrivateFieldDestructureSet.js
|   |   |   |   |   |   |-- ✅ classStaticPrivateFieldDestructureSet.js.map
|   |   |   |   |   |   |-- ✅ classStaticPrivateFieldSpecGet.js
|   |   |   |   |   |   |-- ✅ classStaticPrivateFieldSpecGet.js.map
|   |   |   |   |   |   |-- ✅ classStaticPrivateFieldSpecSet.js
|   |   |   |   |   |   |-- ✅ classStaticPrivateFieldSpecSet.js.map
|   |   |   |   |   |   |-- ✅ classStaticPrivateMethodGet.js
|   |   |   |   |   |   |-- ✅ classStaticPrivateMethodGet.js.map
|   |   |   |   |   |   |-- ✅ classStaticPrivateMethodSet.js
|   |   |   |   |   |   |-- ✅ classStaticPrivateMethodSet.js.map
|   |   |   |   |   |   |-- ✅ construct.js
|   |   |   |   |   |   |-- ✅ construct.js.map
|   |   |   |   |   |   |-- ✅ createClass.js
|   |   |   |   |   |   |-- ✅ createClass.js.map
|   |   |   |   |   |   |-- ✅ createForOfIteratorHelper.js
|   |   |   |   |   |   |-- ✅ createForOfIteratorHelper.js.map
|   |   |   |   |   |   |-- ✅ createForOfIteratorHelperLoose.js
|   |   |   |   |   |   |-- ✅ createForOfIteratorHelperLoose.js.map
|   |   |   |   |   |   |-- ✅ createSuper.js
|   |   |   |   |   |   |-- ✅ createSuper.js.map
|   |   |   |   |   |   |-- ✅ decorate.js
|   |   |   |   |   |   |-- ✅ decorate.js.map
|   |   |   |   |   |   |-- ✅ defaults.js
|   |   |   |   |   |   |-- ✅ defaults.js.map
|   |   |   |   |   |   |-- ✅ defineAccessor.js
|   |   |   |   |   |   |-- ✅ defineAccessor.js.map
|   |   |   |   |   |   |-- ✅ defineEnumerableProperties.js
|   |   |   |   |   |   |-- ✅ defineEnumerableProperties.js.map
|   |   |   |   |   |   |-- ✅ defineProperty.js
|   |   |   |   |   |   |-- ✅ defineProperty.js.map
|   |   |   |   |   |   |-- ✅ dispose.js
|   |   |   |   |   |   |-- ✅ dispose.js.map
|   |   |   |   |   |   |-- ✅ extends.js
|   |   |   |   |   |   |-- ✅ extends.js.map
|   |   |   |   |   |   |-- ✅ get.js
|   |   |   |   |   |   |-- ✅ get.js.map
|   |   |   |   |   |   |-- ✅ getPrototypeOf.js
|   |   |   |   |   |   |-- ✅ getPrototypeOf.js.map
|   |   |   |   |   |   |-- ✅ identity.js
|   |   |   |   |   |   |-- ✅ identity.js.map
|   |   |   |   |   |   |-- ✅ importDeferProxy.js
|   |   |   |   |   |   |-- ✅ importDeferProxy.js.map
|   |   |   |   |   |   |-- ✅ inherits.js
|   |   |   |   |   |   |-- ✅ inherits.js.map
|   |   |   |   |   |   |-- ✅ inheritsLoose.js
|   |   |   |   |   |   |-- ✅ inheritsLoose.js.map
|   |   |   |   |   |   |-- ✅ initializerDefineProperty.js
|   |   |   |   |   |   |-- ✅ initializerDefineProperty.js.map
|   |   |   |   |   |   |-- ✅ initializerWarningHelper.js
|   |   |   |   |   |   |-- ✅ initializerWarningHelper.js.map
|   |   |   |   |   |   |-- ✅ instanceof.js
|   |   |   |   |   |   |-- ✅ instanceof.js.map
|   |   |   |   |   |   |-- ✅ interopRequireDefault.js
|   |   |   |   |   |   |-- ✅ interopRequireDefault.js.map
|   |   |   |   |   |   |-- ✅ interopRequireWildcard.js
|   |   |   |   |   |   |-- ✅ interopRequireWildcard.js.map
|   |   |   |   |   |   |-- ✅ isNativeFunction.js
|   |   |   |   |   |   |-- ✅ isNativeFunction.js.map
|   |   |   |   |   |   |-- ✅ isNativeReflectConstruct.js
|   |   |   |   |   |   |-- ✅ isNativeReflectConstruct.js.map
|   |   |   |   |   |   |-- ✅ iterableToArray.js
|   |   |   |   |   |   |-- ✅ iterableToArray.js.map
|   |   |   |   |   |   |-- ✅ iterableToArrayLimit.js
|   |   |   |   |   |   |-- ✅ iterableToArrayLimit.js.map
|   |   |   |   |   |   |-- ✅ jsx.js
|   |   |   |   |   |   |-- ✅ jsx.js.map
|   |   |   |   |   |   |-- ✅ maybeArrayLike.js
|   |   |   |   |   |   |-- ✅ maybeArrayLike.js.map
|   |   |   |   |   |   |-- ✅ newArrowCheck.js
|   |   |   |   |   |   |-- ✅ newArrowCheck.js.map
|   |   |   |   |   |   |-- ✅ nonIterableRest.js
|   |   |   |   |   |   |-- ✅ nonIterableRest.js.map
|   |   |   |   |   |   |-- ✅ nonIterableSpread.js
|   |   |   |   |   |   |-- ✅ nonIterableSpread.js.map
|   |   |   |   |   |   |-- ✅ nullishReceiverError.js
|   |   |   |   |   |   |-- ✅ nullishReceiverError.js.map
|   |   |   |   |   |   |-- ✅ objectDestructuringEmpty.js
|   |   |   |   |   |   |-- ✅ objectDestructuringEmpty.js.map
|   |   |   |   |   |   |-- ✅ objectSpread.js
|   |   |   |   |   |   |-- ✅ objectSpread.js.map
|   |   |   |   |   |   |-- ✅ objectSpread2.js
|   |   |   |   |   |   |-- ✅ objectSpread2.js.map
|   |   |   |   |   |   |-- ✅ objectWithoutProperties.js
|   |   |   |   |   |   |-- ✅ objectWithoutProperties.js.map
|   |   |   |   |   |   |-- ✅ objectWithoutPropertiesLoose.js
|   |   |   |   |   |   |-- ✅ objectWithoutPropertiesLoose.js.map
|   |   |   |   |   |   |-- ✅ OverloadYield.js
|   |   |   |   |   |   |-- ✅ OverloadYield.js.map
|   |   |   |   |   |   |-- ✅ possibleConstructorReturn.js
|   |   |   |   |   |   |-- ✅ possibleConstructorReturn.js.map
|   |   |   |   |   |   |-- ✅ readOnlyError.js
|   |   |   |   |   |   |-- ✅ readOnlyError.js.map
|   |   |   |   |   |   |-- ✅ regenerator.js
|   |   |   |   |   |   |-- ✅ regenerator.js.map
|   |   |   |   |   |   |-- ✅ regeneratorAsync.js
|   |   |   |   |   |   |-- ✅ regeneratorAsync.js.map
|   |   |   |   |   |   |-- ✅ regeneratorAsyncGen.js
|   |   |   |   |   |   |-- ✅ regeneratorAsyncGen.js.map
|   |   |   |   |   |   |-- ✅ regeneratorAsyncIterator.js
|   |   |   |   |   |   |-- ✅ regeneratorAsyncIterator.js.map
|   |   |   |   |   |   |-- ✅ regeneratorDefine.js
|   |   |   |   |   |   |-- ✅ regeneratorDefine.js.map
|   |   |   |   |   |   |-- ✅ regeneratorKeys.js
|   |   |   |   |   |   |-- ✅ regeneratorKeys.js.map
|   |   |   |   |   |   |-- ✅ regeneratorRuntime.js
|   |   |   |   |   |   |-- ✅ regeneratorRuntime.js.map
|   |   |   |   |   |   |-- ✅ regeneratorValues.js
|   |   |   |   |   |   |-- ✅ regeneratorValues.js.map
|   |   |   |   |   |   |-- ✅ set.js
|   |   |   |   |   |   |-- ✅ set.js.map
|   |   |   |   |   |   |-- ✅ setFunctionName.js
|   |   |   |   |   |   |-- ✅ setFunctionName.js.map
|   |   |   |   |   |   |-- ✅ setPrototypeOf.js
|   |   |   |   |   |   |-- ✅ setPrototypeOf.js.map
|   |   |   |   |   |   |-- ✅ skipFirstGeneratorNext.js
|   |   |   |   |   |   |-- ✅ skipFirstGeneratorNext.js.map
|   |   |   |   |   |   |-- ✅ slicedToArray.js
|   |   |   |   |   |   |-- ✅ slicedToArray.js.map
|   |   |   |   |   |   |-- ✅ superPropBase.js
|   |   |   |   |   |   |-- ✅ superPropBase.js.map
|   |   |   |   |   |   |-- ✅ superPropGet.js
|   |   |   |   |   |   |-- ✅ superPropGet.js.map
|   |   |   |   |   |   |-- ✅ superPropSet.js
|   |   |   |   |   |   |-- ✅ superPropSet.js.map
|   |   |   |   |   |   |-- ✅ taggedTemplateLiteral.js
|   |   |   |   |   |   |-- ✅ taggedTemplateLiteral.js.map
|   |   |   |   |   |   |-- ✅ taggedTemplateLiteralLoose.js
|   |   |   |   |   |   |-- ✅ taggedTemplateLiteralLoose.js.map
|   |   |   |   |   |   |-- ✅ tdz.js
|   |   |   |   |   |   |-- ✅ tdz.js.map
|   |   |   |   |   |   |-- ✅ temporalRef.js
|   |   |   |   |   |   |-- ✅ temporalRef.js.map
|   |   |   |   |   |   |-- ✅ temporalUndefined.js
|   |   |   |   |   |   |-- ✅ temporalUndefined.js.map
|   |   |   |   |   |   |-- ✅ toArray.js
|   |   |   |   |   |   |-- ✅ toArray.js.map
|   |   |   |   |   |   |-- ✅ toConsumableArray.js
|   |   |   |   |   |   |-- ✅ toConsumableArray.js.map
|   |   |   |   |   |   |-- ✅ toPrimitive.js
|   |   |   |   |   |   |-- ✅ toPrimitive.js.map
|   |   |   |   |   |   |-- ✅ toPropertyKey.js
|   |   |   |   |   |   |-- ✅ toPropertyKey.js.map
|   |   |   |   |   |   |-- ✅ toSetter.js
|   |   |   |   |   |   |-- ✅ toSetter.js.map
|   |   |   |   |   |   |-- ✅ tsRewriteRelativeImportExtensions.js
|   |   |   |   |   |   |-- ✅ tsRewriteRelativeImportExtensions.js.map
|   |   |   |   |   |   |-- ✅ typeof.js
|   |   |   |   |   |   |-- ✅ typeof.js.map
|   |   |   |   |   |   |-- ✅ unsupportedIterableToArray.js
|   |   |   |   |   |   |-- ✅ unsupportedIterableToArray.js.map
|   |   |   |   |   |   |-- ✅ using.js
|   |   |   |   |   |   |-- ✅ using.js.map
|   |   |   |   |   |   |-- ✅ usingCtx.js
|   |   |   |   |   |   |-- ✅ usingCtx.js.map
|   |   |   |   |   |   |-- ✅ wrapAsyncGenerator.js
|   |   |   |   |   |   |-- ✅ wrapAsyncGenerator.js.map
|   |   |   |   |   |   |-- ✅ wrapNativeSuper.js
|   |   |   |   |   |   |-- ✅ wrapNativeSuper.js.map
|   |   |   |   |   |   |-- ✅ wrapRegExp.js
|   |   |   |   |   |   |-- ✅ wrapRegExp.js.map
|   |   |   |   |   |   |-- ✅ writeOnlyError.js
|   |   |   |   |   |   \-- ✅ writeOnlyError.js.map
|   |   |   |   |   |-- ✅ helpers-generated.js
|   |   |   |   |   |-- ✅ helpers-generated.js.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   \-- ✅ index.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ parser/
|   |   |   |   |-- ✅ bin/
|   |   |   |   |   \-- ✅ babel-parser.js
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   \-- ✅ index.js.map
|   |   |   |   |-- ✅ typings/
|   |   |   |   |   \-- ✅ babel-parser.d.ts
|   |   |   |   |-- ✅ CHANGELOG.md
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ plugin-transform-react-jsx-self/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   \-- ✅ index.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ plugin-transform-react-jsx-source/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   \-- ✅ index.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ runtime/
|   |   |   |   |-- ✅ helpers/
|   |   |   |   |   |-- ✅ esm/
|   |   |   |   |   |   |-- ✅ applyDecoratedDescriptor.js
|   |   |   |   |   |   |-- ✅ applyDecs.js
|   |   |   |   |   |   |-- ✅ applyDecs2203.js
|   |   |   |   |   |   |-- ✅ applyDecs2203R.js
|   |   |   |   |   |   |-- ✅ applyDecs2301.js
|   |   |   |   |   |   |-- ✅ applyDecs2305.js
|   |   |   |   |   |   |-- ✅ applyDecs2311.js
|   |   |   |   |   |   |-- ✅ arrayLikeToArray.js
|   |   |   |   |   |   |-- ✅ arrayWithHoles.js
|   |   |   |   |   |   |-- ✅ arrayWithoutHoles.js
|   |   |   |   |   |   |-- ✅ assertClassBrand.js
|   |   |   |   |   |   |-- ✅ assertThisInitialized.js
|   |   |   |   |   |   |-- ✅ asyncGeneratorDelegate.js
|   |   |   |   |   |   |-- ✅ asyncIterator.js
|   |   |   |   |   |   |-- ✅ asyncToGenerator.js
|   |   |   |   |   |   |-- ✅ awaitAsyncGenerator.js
|   |   |   |   |   |   |-- ✅ AwaitValue.js
|   |   |   |   |   |   |-- ✅ callSuper.js
|   |   |   |   |   |   |-- ✅ checkInRHS.js
|   |   |   |   |   |   |-- ✅ checkPrivateRedeclaration.js
|   |   |   |   |   |   |-- ✅ classApplyDescriptorDestructureSet.js
|   |   |   |   |   |   |-- ✅ classApplyDescriptorGet.js
|   |   |   |   |   |   |-- ✅ classApplyDescriptorSet.js
|   |   |   |   |   |   |-- ✅ classCallCheck.js
|   |   |   |   |   |   |-- ✅ classCheckPrivateStaticAccess.js
|   |   |   |   |   |   |-- ✅ classCheckPrivateStaticFieldDescriptor.js
|   |   |   |   |   |   |-- ✅ classExtractFieldDescriptor.js
|   |   |   |   |   |   |-- ✅ classNameTDZError.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldDestructureSet.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldGet.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldGet2.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldInitSpec.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldLooseBase.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldLooseKey.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldSet.js
|   |   |   |   |   |   |-- ✅ classPrivateFieldSet2.js
|   |   |   |   |   |   |-- ✅ classPrivateGetter.js
|   |   |   |   |   |   |-- ✅ classPrivateMethodGet.js
|   |   |   |   |   |   |-- ✅ classPrivateMethodInitSpec.js
|   |   |   |   |   |   |-- ✅ classPrivateMethodSet.js
|   |   |   |   |   |   |-- ✅ classPrivateSetter.js
|   |   |   |   |   |   |-- ✅ classStaticPrivateFieldDestructureSet.js
|   |   |   |   |   |   |-- ✅ classStaticPrivateFieldSpecGet.js
|   |   |   |   |   |   |-- ✅ classStaticPrivateFieldSpecSet.js
|   |   |   |   |   |   |-- ✅ classStaticPrivateMethodGet.js
|   |   |   |   |   |   |-- ✅ classStaticPrivateMethodSet.js
|   |   |   |   |   |   |-- ✅ construct.js
|   |   |   |   |   |   |-- ✅ createClass.js
|   |   |   |   |   |   |-- ✅ createForOfIteratorHelper.js
|   |   |   |   |   |   |-- ✅ createForOfIteratorHelperLoose.js
|   |   |   |   |   |   |-- ✅ createSuper.js
|   |   |   |   |   |   |-- ✅ decorate.js
|   |   |   |   |   |   |-- ✅ defaults.js
|   |   |   |   |   |   |-- ✅ defineAccessor.js
|   |   |   |   |   |   |-- ✅ defineEnumerableProperties.js
|   |   |   |   |   |   |-- ✅ defineProperty.js
|   |   |   |   |   |   |-- ✅ dispose.js
|   |   |   |   |   |   |-- ✅ extends.js
|   |   |   |   |   |   |-- ✅ get.js
|   |   |   |   |   |   |-- ✅ getPrototypeOf.js
|   |   |   |   |   |   |-- ✅ identity.js
|   |   |   |   |   |   |-- ✅ importDeferProxy.js
|   |   |   |   |   |   |-- ✅ inherits.js
|   |   |   |   |   |   |-- ✅ inheritsLoose.js
|   |   |   |   |   |   |-- ✅ initializerDefineProperty.js
|   |   |   |   |   |   |-- ✅ initializerWarningHelper.js
|   |   |   |   |   |   |-- ✅ instanceof.js
|   |   |   |   |   |   |-- ✅ interopRequireDefault.js
|   |   |   |   |   |   |-- ✅ interopRequireWildcard.js
|   |   |   |   |   |   |-- ✅ isNativeFunction.js
|   |   |   |   |   |   |-- ✅ isNativeReflectConstruct.js
|   |   |   |   |   |   |-- ✅ iterableToArray.js
|   |   |   |   |   |   |-- ✅ iterableToArrayLimit.js
|   |   |   |   |   |   |-- ✅ jsx.js
|   |   |   |   |   |   |-- ✅ maybeArrayLike.js
|   |   |   |   |   |   |-- ✅ newArrowCheck.js
|   |   |   |   |   |   |-- ✅ nonIterableRest.js
|   |   |   |   |   |   |-- ✅ nonIterableSpread.js
|   |   |   |   |   |   |-- ✅ nullishReceiverError.js
|   |   |   |   |   |   |-- ✅ objectDestructuringEmpty.js
|   |   |   |   |   |   |-- ✅ objectSpread.js
|   |   |   |   |   |   |-- ✅ objectSpread2.js
|   |   |   |   |   |   |-- ✅ objectWithoutProperties.js
|   |   |   |   |   |   |-- ✅ objectWithoutPropertiesLoose.js
|   |   |   |   |   |   |-- ✅ OverloadYield.js
|   |   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |   |-- ✅ possibleConstructorReturn.js
|   |   |   |   |   |   |-- ✅ readOnlyError.js
|   |   |   |   |   |   |-- ✅ regenerator.js
|   |   |   |   |   |   |-- ✅ regeneratorAsync.js
|   |   |   |   |   |   |-- ✅ regeneratorAsyncGen.js
|   |   |   |   |   |   |-- ✅ regeneratorAsyncIterator.js
|   |   |   |   |   |   |-- ✅ regeneratorDefine.js
|   |   |   |   |   |   |-- ✅ regeneratorKeys.js
|   |   |   |   |   |   |-- ✅ regeneratorRuntime.js
|   |   |   |   |   |   |-- ✅ regeneratorValues.js
|   |   |   |   |   |   |-- ✅ set.js
|   |   |   |   |   |   |-- ✅ setFunctionName.js
|   |   |   |   |   |   |-- ✅ setPrototypeOf.js
|   |   |   |   |   |   |-- ✅ skipFirstGeneratorNext.js
|   |   |   |   |   |   |-- ✅ slicedToArray.js
|   |   |   |   |   |   |-- ✅ superPropBase.js
|   |   |   |   |   |   |-- ✅ superPropGet.js
|   |   |   |   |   |   |-- ✅ superPropSet.js
|   |   |   |   |   |   |-- ✅ taggedTemplateLiteral.js
|   |   |   |   |   |   |-- ✅ taggedTemplateLiteralLoose.js
|   |   |   |   |   |   |-- ✅ tdz.js
|   |   |   |   |   |   |-- ✅ temporalRef.js
|   |   |   |   |   |   |-- ✅ temporalUndefined.js
|   |   |   |   |   |   |-- ✅ toArray.js
|   |   |   |   |   |   |-- ✅ toConsumableArray.js
|   |   |   |   |   |   |-- ✅ toPrimitive.js
|   |   |   |   |   |   |-- ✅ toPropertyKey.js
|   |   |   |   |   |   |-- ✅ toSetter.js
|   |   |   |   |   |   |-- ✅ tsRewriteRelativeImportExtensions.js
|   |   |   |   |   |   |-- ✅ typeof.js
|   |   |   |   |   |   |-- ✅ unsupportedIterableToArray.js
|   |   |   |   |   |   |-- ✅ using.js
|   |   |   |   |   |   |-- ✅ usingCtx.js
|   |   |   |   |   |   |-- ✅ wrapAsyncGenerator.js
|   |   |   |   |   |   |-- ✅ wrapNativeSuper.js
|   |   |   |   |   |   |-- ✅ wrapRegExp.js
|   |   |   |   |   |   \-- ✅ writeOnlyError.js
|   |   |   |   |   |-- ✅ applyDecoratedDescriptor.js
|   |   |   |   |   |-- ✅ applyDecs.js
|   |   |   |   |   |-- ✅ applyDecs2203.js
|   |   |   |   |   |-- ✅ applyDecs2203R.js
|   |   |   |   |   |-- ✅ applyDecs2301.js
|   |   |   |   |   |-- ✅ applyDecs2305.js
|   |   |   |   |   |-- ✅ applyDecs2311.js
|   |   |   |   |   |-- ✅ arrayLikeToArray.js
|   |   |   |   |   |-- ✅ arrayWithHoles.js
|   |   |   |   |   |-- ✅ arrayWithoutHoles.js
|   |   |   |   |   |-- ✅ assertClassBrand.js
|   |   |   |   |   |-- ✅ assertThisInitialized.js
|   |   |   |   |   |-- ✅ asyncGeneratorDelegate.js
|   |   |   |   |   |-- ✅ asyncIterator.js
|   |   |   |   |   |-- ✅ asyncToGenerator.js
|   |   |   |   |   |-- ✅ awaitAsyncGenerator.js
|   |   |   |   |   |-- ✅ AwaitValue.js
|   |   |   |   |   |-- ✅ callSuper.js
|   |   |   |   |   |-- ✅ checkInRHS.js
|   |   |   |   |   |-- ✅ checkPrivateRedeclaration.js
|   |   |   |   |   |-- ✅ classApplyDescriptorDestructureSet.js
|   |   |   |   |   |-- ✅ classApplyDescriptorGet.js
|   |   |   |   |   |-- ✅ classApplyDescriptorSet.js
|   |   |   |   |   |-- ✅ classCallCheck.js
|   |   |   |   |   |-- ✅ classCheckPrivateStaticAccess.js
|   |   |   |   |   |-- ✅ classCheckPrivateStaticFieldDescriptor.js
|   |   |   |   |   |-- ✅ classExtractFieldDescriptor.js
|   |   |   |   |   |-- ✅ classNameTDZError.js
|   |   |   |   |   |-- ✅ classPrivateFieldDestructureSet.js
|   |   |   |   |   |-- ✅ classPrivateFieldGet.js
|   |   |   |   |   |-- ✅ classPrivateFieldGet2.js
|   |   |   |   |   |-- ✅ classPrivateFieldInitSpec.js
|   |   |   |   |   |-- ✅ classPrivateFieldLooseBase.js
|   |   |   |   |   |-- ✅ classPrivateFieldLooseKey.js
|   |   |   |   |   |-- ✅ classPrivateFieldSet.js
|   |   |   |   |   |-- ✅ classPrivateFieldSet2.js
|   |   |   |   |   |-- ✅ classPrivateGetter.js
|   |   |   |   |   |-- ✅ classPrivateMethodGet.js
|   |   |   |   |   |-- ✅ classPrivateMethodInitSpec.js
|   |   |   |   |   |-- ✅ classPrivateMethodSet.js
|   |   |   |   |   |-- ✅ classPrivateSetter.js
|   |   |   |   |   |-- ✅ classStaticPrivateFieldDestructureSet.js
|   |   |   |   |   |-- ✅ classStaticPrivateFieldSpecGet.js
|   |   |   |   |   |-- ✅ classStaticPrivateFieldSpecSet.js
|   |   |   |   |   |-- ✅ classStaticPrivateMethodGet.js
|   |   |   |   |   |-- ✅ classStaticPrivateMethodSet.js
|   |   |   |   |   |-- ✅ construct.js
|   |   |   |   |   |-- ✅ createClass.js
|   |   |   |   |   |-- ✅ createForOfIteratorHelper.js
|   |   |   |   |   |-- ✅ createForOfIteratorHelperLoose.js
|   |   |   |   |   |-- ✅ createSuper.js
|   |   |   |   |   |-- ✅ decorate.js
|   |   |   |   |   |-- ✅ defaults.js
|   |   |   |   |   |-- ✅ defineAccessor.js
|   |   |   |   |   |-- ✅ defineEnumerableProperties.js
|   |   |   |   |   |-- ✅ defineProperty.js
|   |   |   |   |   |-- ✅ dispose.js
|   |   |   |   |   |-- ✅ extends.js
|   |   |   |   |   |-- ✅ get.js
|   |   |   |   |   |-- ✅ getPrototypeOf.js
|   |   |   |   |   |-- ✅ identity.js
|   |   |   |   |   |-- ✅ importDeferProxy.js
|   |   |   |   |   |-- ✅ inherits.js
|   |   |   |   |   |-- ✅ inheritsLoose.js
|   |   |   |   |   |-- ✅ initializerDefineProperty.js
|   |   |   |   |   |-- ✅ initializerWarningHelper.js
|   |   |   |   |   |-- ✅ instanceof.js
|   |   |   |   |   |-- ✅ interopRequireDefault.js
|   |   |   |   |   |-- ✅ interopRequireWildcard.js
|   |   |   |   |   |-- ✅ isNativeFunction.js
|   |   |   |   |   |-- ✅ isNativeReflectConstruct.js
|   |   |   |   |   |-- ✅ iterableToArray.js
|   |   |   |   |   |-- ✅ iterableToArrayLimit.js
|   |   |   |   |   |-- ✅ jsx.js
|   |   |   |   |   |-- ✅ maybeArrayLike.js
|   |   |   |   |   |-- ✅ newArrowCheck.js
|   |   |   |   |   |-- ✅ nonIterableRest.js
|   |   |   |   |   |-- ✅ nonIterableSpread.js
|   |   |   |   |   |-- ✅ nullishReceiverError.js
|   |   |   |   |   |-- ✅ objectDestructuringEmpty.js
|   |   |   |   |   |-- ✅ objectSpread.js
|   |   |   |   |   |-- ✅ objectSpread2.js
|   |   |   |   |   |-- ✅ objectWithoutProperties.js
|   |   |   |   |   |-- ✅ objectWithoutPropertiesLoose.js
|   |   |   |   |   |-- ✅ OverloadYield.js
|   |   |   |   |   |-- ✅ possibleConstructorReturn.js
|   |   |   |   |   |-- ✅ readOnlyError.js
|   |   |   |   |   |-- ✅ regenerator.js
|   |   |   |   |   |-- ✅ regeneratorAsync.js
|   |   |   |   |   |-- ✅ regeneratorAsyncGen.js
|   |   |   |   |   |-- ✅ regeneratorAsyncIterator.js
|   |   |   |   |   |-- ✅ regeneratorDefine.js
|   |   |   |   |   |-- ✅ regeneratorKeys.js
|   |   |   |   |   |-- ✅ regeneratorRuntime.js
|   |   |   |   |   |-- ✅ regeneratorValues.js
|   |   |   |   |   |-- ✅ set.js
|   |   |   |   |   |-- ✅ setFunctionName.js
|   |   |   |   |   |-- ✅ setPrototypeOf.js
|   |   |   |   |   |-- ✅ skipFirstGeneratorNext.js
|   |   |   |   |   |-- ✅ slicedToArray.js
|   |   |   |   |   |-- ✅ superPropBase.js
|   |   |   |   |   |-- ✅ superPropGet.js
|   |   |   |   |   |-- ✅ superPropSet.js
|   |   |   |   |   |-- ✅ taggedTemplateLiteral.js
|   |   |   |   |   |-- ✅ taggedTemplateLiteralLoose.js
|   |   |   |   |   |-- ✅ tdz.js
|   |   |   |   |   |-- ✅ temporalRef.js
|   |   |   |   |   |-- ✅ temporalUndefined.js
|   |   |   |   |   |-- ✅ toArray.js
|   |   |   |   |   |-- ✅ toConsumableArray.js
|   |   |   |   |   |-- ✅ toPrimitive.js
|   |   |   |   |   |-- ✅ toPropertyKey.js
|   |   |   |   |   |-- ✅ toSetter.js
|   |   |   |   |   |-- ✅ tsRewriteRelativeImportExtensions.js
|   |   |   |   |   |-- ✅ typeof.js
|   |   |   |   |   |-- ✅ unsupportedIterableToArray.js
|   |   |   |   |   |-- ✅ using.js
|   |   |   |   |   |-- ✅ usingCtx.js
|   |   |   |   |   |-- ✅ wrapAsyncGenerator.js
|   |   |   |   |   |-- ✅ wrapNativeSuper.js
|   |   |   |   |   |-- ✅ wrapRegExp.js
|   |   |   |   |   \-- ✅ writeOnlyError.js
|   |   |   |   |-- ✅ regenerator/
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ template/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ builder.js
|   |   |   |   |   |-- ✅ builder.js.map
|   |   |   |   |   |-- ✅ formatters.js
|   |   |   |   |   |-- ✅ formatters.js.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ literal.js
|   |   |   |   |   |-- ✅ literal.js.map
|   |   |   |   |   |-- ✅ options.js
|   |   |   |   |   |-- ✅ options.js.map
|   |   |   |   |   |-- ✅ parse.js
|   |   |   |   |   |-- ✅ parse.js.map
|   |   |   |   |   |-- ✅ populate.js
|   |   |   |   |   |-- ✅ populate.js.map
|   |   |   |   |   |-- ✅ string.js
|   |   |   |   |   \-- ✅ string.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ traverse/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ path/
|   |   |   |   |   |   |-- ✅ inference/
|   |   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |   |   |-- ✅ inferer-reference.js
|   |   |   |   |   |   |   |-- ✅ inferer-reference.js.map
|   |   |   |   |   |   |   |-- ✅ inferers.js
|   |   |   |   |   |   |   |-- ✅ inferers.js.map
|   |   |   |   |   |   |   |-- ✅ util.js
|   |   |   |   |   |   |   \-- ✅ util.js.map
|   |   |   |   |   |   |-- ✅ lib/
|   |   |   |   |   |   |   |-- ✅ hoister.js
|   |   |   |   |   |   |   |-- ✅ hoister.js.map
|   |   |   |   |   |   |   |-- ✅ removal-hooks.js
|   |   |   |   |   |   |   |-- ✅ removal-hooks.js.map
|   |   |   |   |   |   |   |-- ✅ virtual-types-validator.js
|   |   |   |   |   |   |   |-- ✅ virtual-types-validator.js.map
|   |   |   |   |   |   |   |-- ✅ virtual-types.js
|   |   |   |   |   |   |   \-- ✅ virtual-types.js.map
|   |   |   |   |   |   |-- ✅ ancestry.js
|   |   |   |   |   |   |-- ✅ ancestry.js.map
|   |   |   |   |   |   |-- ✅ comments.js
|   |   |   |   |   |   |-- ✅ comments.js.map
|   |   |   |   |   |   |-- ✅ context.js
|   |   |   |   |   |   |-- ✅ context.js.map
|   |   |   |   |   |   |-- ✅ conversion.js
|   |   |   |   |   |   |-- ✅ conversion.js.map
|   |   |   |   |   |   |-- ✅ evaluation.js
|   |   |   |   |   |   |-- ✅ evaluation.js.map
|   |   |   |   |   |   |-- ✅ family.js
|   |   |   |   |   |   |-- ✅ family.js.map
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |   |-- ✅ introspection.js
|   |   |   |   |   |   |-- ✅ introspection.js.map
|   |   |   |   |   |   |-- ✅ modification.js
|   |   |   |   |   |   |-- ✅ modification.js.map
|   |   |   |   |   |   |-- ✅ removal.js
|   |   |   |   |   |   |-- ✅ removal.js.map
|   |   |   |   |   |   |-- ✅ replacement.js
|   |   |   |   |   |   \-- ✅ replacement.js.map
|   |   |   |   |   |-- ✅ scope/
|   |   |   |   |   |   |-- ✅ lib/
|   |   |   |   |   |   |   |-- ✅ renamer.js
|   |   |   |   |   |   |   \-- ✅ renamer.js.map
|   |   |   |   |   |   |-- ✅ binding.js
|   |   |   |   |   |   |-- ✅ binding.js.map
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ index.js.map
|   |   |   |   |   |-- ✅ cache.js
|   |   |   |   |   |-- ✅ cache.js.map
|   |   |   |   |   |-- ✅ context.js
|   |   |   |   |   |-- ✅ context.js.map
|   |   |   |   |   |-- ✅ hub.js
|   |   |   |   |   |-- ✅ hub.js.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ traverse-node.js
|   |   |   |   |   |-- ✅ traverse-node.js.map
|   |   |   |   |   |-- ✅ types.js
|   |   |   |   |   |-- ✅ types.js.map
|   |   |   |   |   |-- ✅ visitors.js
|   |   |   |   |   \-- ✅ visitors.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   \-- ✅ types/
|   |   |       |-- ✅ lib/
|   |   |       |   |-- ✅ asserts/
|   |   |       |   |   |-- ✅ generated/
|   |   |       |   |   |   |-- ✅ index.js
|   |   |       |   |   |   \-- ✅ index.js.map
|   |   |       |   |   |-- ✅ assertNode.js
|   |   |       |   |   \-- ✅ assertNode.js.map
|   |   |       |   |-- ✅ ast-types/
|   |   |       |   |   \-- ✅ generated/
|   |   |       |   |       |-- ✅ index.js
|   |   |       |   |       \-- ✅ index.js.map
|   |   |       |   |-- ✅ builders/
|   |   |       |   |   |-- ✅ flow/
|   |   |       |   |   |   |-- ✅ createFlowUnionType.js
|   |   |       |   |   |   |-- ✅ createFlowUnionType.js.map
|   |   |       |   |   |   |-- ✅ createTypeAnnotationBasedOnTypeof.js
|   |   |       |   |   |   \-- ✅ createTypeAnnotationBasedOnTypeof.js.map
|   |   |       |   |   |-- ✅ generated/
|   |   |       |   |   |   |-- ✅ index.js
|   |   |       |   |   |   |-- ✅ index.js.map
|   |   |       |   |   |   |-- ✅ lowercase.js
|   |   |       |   |   |   |-- ✅ lowercase.js.map
|   |   |       |   |   |   |-- ✅ uppercase.js
|   |   |       |   |   |   \-- ✅ uppercase.js.map
|   |   |       |   |   |-- ✅ react/
|   |   |       |   |   |   |-- ✅ buildChildren.js
|   |   |       |   |   |   \-- ✅ buildChildren.js.map
|   |   |       |   |   |-- ✅ typescript/
|   |   |       |   |   |   |-- ✅ createTSUnionType.js
|   |   |       |   |   |   \-- ✅ createTSUnionType.js.map
|   |   |       |   |   |-- ✅ productions.js
|   |   |       |   |   |-- ✅ productions.js.map
|   |   |       |   |   |-- ✅ validateNode.js
|   |   |       |   |   \-- ✅ validateNode.js.map
|   |   |       |   |-- ✅ clone/
|   |   |       |   |   |-- ✅ clone.js
|   |   |       |   |   |-- ✅ clone.js.map
|   |   |       |   |   |-- ✅ cloneDeep.js
|   |   |       |   |   |-- ✅ cloneDeep.js.map
|   |   |       |   |   |-- ✅ cloneDeepWithoutLoc.js
|   |   |       |   |   |-- ✅ cloneDeepWithoutLoc.js.map
|   |   |       |   |   |-- ✅ cloneNode.js
|   |   |       |   |   |-- ✅ cloneNode.js.map
|   |   |       |   |   |-- ✅ cloneWithoutLoc.js
|   |   |       |   |   \-- ✅ cloneWithoutLoc.js.map
|   |   |       |   |-- ✅ comments/
|   |   |       |   |   |-- ✅ addComment.js
|   |   |       |   |   |-- ✅ addComment.js.map
|   |   |       |   |   |-- ✅ addComments.js
|   |   |       |   |   |-- ✅ addComments.js.map
|   |   |       |   |   |-- ✅ inheritInnerComments.js
|   |   |       |   |   |-- ✅ inheritInnerComments.js.map
|   |   |       |   |   |-- ✅ inheritLeadingComments.js
|   |   |       |   |   |-- ✅ inheritLeadingComments.js.map
|   |   |       |   |   |-- ✅ inheritsComments.js
|   |   |       |   |   |-- ✅ inheritsComments.js.map
|   |   |       |   |   |-- ✅ inheritTrailingComments.js
|   |   |       |   |   |-- ✅ inheritTrailingComments.js.map
|   |   |       |   |   |-- ✅ removeComments.js
|   |   |       |   |   \-- ✅ removeComments.js.map
|   |   |       |   |-- ✅ constants/
|   |   |       |   |   |-- ✅ generated/
|   |   |       |   |   |   |-- ✅ index.js
|   |   |       |   |   |   \-- ✅ index.js.map
|   |   |       |   |   |-- ✅ index.js
|   |   |       |   |   \-- ✅ index.js.map
|   |   |       |   |-- ✅ converters/
|   |   |       |   |   |-- ✅ ensureBlock.js
|   |   |       |   |   |-- ✅ ensureBlock.js.map
|   |   |       |   |   |-- ✅ gatherSequenceExpressions.js
|   |   |       |   |   |-- ✅ gatherSequenceExpressions.js.map
|   |   |       |   |   |-- ✅ toBindingIdentifierName.js
|   |   |       |   |   |-- ✅ toBindingIdentifierName.js.map
|   |   |       |   |   |-- ✅ toBlock.js
|   |   |       |   |   |-- ✅ toBlock.js.map
|   |   |       |   |   |-- ✅ toComputedKey.js
|   |   |       |   |   |-- ✅ toComputedKey.js.map
|   |   |       |   |   |-- ✅ toExpression.js
|   |   |       |   |   |-- ✅ toExpression.js.map
|   |   |       |   |   |-- ✅ toIdentifier.js
|   |   |       |   |   |-- ✅ toIdentifier.js.map
|   |   |       |   |   |-- ✅ toKeyAlias.js
|   |   |       |   |   |-- ✅ toKeyAlias.js.map
|   |   |       |   |   |-- ✅ toSequenceExpression.js
|   |   |       |   |   |-- ✅ toSequenceExpression.js.map
|   |   |       |   |   |-- ✅ toStatement.js
|   |   |       |   |   |-- ✅ toStatement.js.map
|   |   |       |   |   |-- ✅ valueToNode.js
|   |   |       |   |   \-- ✅ valueToNode.js.map
|   |   |       |   |-- ✅ definitions/
|   |   |       |   |   |-- ✅ core.js
|   |   |       |   |   |-- ✅ core.js.map
|   |   |       |   |   |-- ✅ deprecated-aliases.js
|   |   |       |   |   |-- ✅ deprecated-aliases.js.map
|   |   |       |   |   |-- ✅ experimental.js
|   |   |       |   |   |-- ✅ experimental.js.map
|   |   |       |   |   |-- ✅ flow.js
|   |   |       |   |   |-- ✅ flow.js.map
|   |   |       |   |   |-- ✅ index.js
|   |   |       |   |   |-- ✅ index.js.map
|   |   |       |   |   |-- ✅ jsx.js
|   |   |       |   |   |-- ✅ jsx.js.map
|   |   |       |   |   |-- ✅ misc.js
|   |   |       |   |   |-- ✅ misc.js.map
|   |   |       |   |   |-- ✅ placeholders.js
|   |   |       |   |   |-- ✅ placeholders.js.map
|   |   |       |   |   |-- ✅ typescript.js
|   |   |       |   |   |-- ✅ typescript.js.map
|   |   |       |   |   |-- ✅ utils.js
|   |   |       |   |   \-- ✅ utils.js.map
|   |   |       |   |-- ✅ modifications/
|   |   |       |   |   |-- ✅ flow/
|   |   |       |   |   |   |-- ✅ removeTypeDuplicates.js
|   |   |       |   |   |   \-- ✅ removeTypeDuplicates.js.map
|   |   |       |   |   |-- ✅ typescript/
|   |   |       |   |   |   |-- ✅ removeTypeDuplicates.js
|   |   |       |   |   |   \-- ✅ removeTypeDuplicates.js.map
|   |   |       |   |   |-- ✅ appendToMemberExpression.js
|   |   |       |   |   |-- ✅ appendToMemberExpression.js.map
|   |   |       |   |   |-- ✅ inherits.js
|   |   |       |   |   |-- ✅ inherits.js.map
|   |   |       |   |   |-- ✅ prependToMemberExpression.js
|   |   |       |   |   |-- ✅ prependToMemberExpression.js.map
|   |   |       |   |   |-- ✅ removeProperties.js
|   |   |       |   |   |-- ✅ removeProperties.js.map
|   |   |       |   |   |-- ✅ removePropertiesDeep.js
|   |   |       |   |   \-- ✅ removePropertiesDeep.js.map
|   |   |       |   |-- ✅ retrievers/
|   |   |       |   |   |-- ✅ getAssignmentIdentifiers.js
|   |   |       |   |   |-- ✅ getAssignmentIdentifiers.js.map
|   |   |       |   |   |-- ✅ getBindingIdentifiers.js
|   |   |       |   |   |-- ✅ getBindingIdentifiers.js.map
|   |   |       |   |   |-- ✅ getFunctionName.js
|   |   |       |   |   |-- ✅ getFunctionName.js.map
|   |   |       |   |   |-- ✅ getOuterBindingIdentifiers.js
|   |   |       |   |   \-- ✅ getOuterBindingIdentifiers.js.map
|   |   |       |   |-- ✅ traverse/
|   |   |       |   |   |-- ✅ traverse.js
|   |   |       |   |   |-- ✅ traverse.js.map
|   |   |       |   |   |-- ✅ traverseFast.js
|   |   |       |   |   \-- ✅ traverseFast.js.map
|   |   |       |   |-- ✅ utils/
|   |   |       |   |   |-- ✅ react/
|   |   |       |   |   |   |-- ✅ cleanJSXElementLiteralChild.js
|   |   |       |   |   |   \-- ✅ cleanJSXElementLiteralChild.js.map
|   |   |       |   |   |-- ✅ deprecationWarning.js
|   |   |       |   |   |-- ✅ deprecationWarning.js.map
|   |   |       |   |   |-- ✅ inherit.js
|   |   |       |   |   |-- ✅ inherit.js.map
|   |   |       |   |   |-- ✅ shallowEqual.js
|   |   |       |   |   \-- ✅ shallowEqual.js.map
|   |   |       |   |-- ✅ validators/
|   |   |       |   |   |-- ✅ generated/
|   |   |       |   |   |   |-- ✅ index.js
|   |   |       |   |   |   \-- ✅ index.js.map
|   |   |       |   |   |-- ✅ react/
|   |   |       |   |   |   |-- ✅ isCompatTag.js
|   |   |       |   |   |   |-- ✅ isCompatTag.js.map
|   |   |       |   |   |   |-- ✅ isReactComponent.js
|   |   |       |   |   |   \-- ✅ isReactComponent.js.map
|   |   |       |   |   |-- ✅ buildMatchMemberExpression.js
|   |   |       |   |   |-- ✅ buildMatchMemberExpression.js.map
|   |   |       |   |   |-- ✅ is.js
|   |   |       |   |   |-- ✅ is.js.map
|   |   |       |   |   |-- ✅ isBinding.js
|   |   |       |   |   |-- ✅ isBinding.js.map
|   |   |       |   |   |-- ✅ isBlockScoped.js
|   |   |       |   |   |-- ✅ isBlockScoped.js.map
|   |   |       |   |   |-- ✅ isImmutable.js
|   |   |       |   |   |-- ✅ isImmutable.js.map
|   |   |       |   |   |-- ✅ isLet.js
|   |   |       |   |   |-- ✅ isLet.js.map
|   |   |       |   |   |-- ✅ isNode.js
|   |   |       |   |   |-- ✅ isNode.js.map
|   |   |       |   |   |-- ✅ isNodesEquivalent.js
|   |   |       |   |   |-- ✅ isNodesEquivalent.js.map
|   |   |       |   |   |-- ✅ isPlaceholderType.js
|   |   |       |   |   |-- ✅ isPlaceholderType.js.map
|   |   |       |   |   |-- ✅ isReferenced.js
|   |   |       |   |   |-- ✅ isReferenced.js.map
|   |   |       |   |   |-- ✅ isScope.js
|   |   |       |   |   |-- ✅ isScope.js.map
|   |   |       |   |   |-- ✅ isSpecifierDefault.js
|   |   |       |   |   |-- ✅ isSpecifierDefault.js.map
|   |   |       |   |   |-- ✅ isType.js
|   |   |       |   |   |-- ✅ isType.js.map
|   |   |       |   |   |-- ✅ isValidES3Identifier.js
|   |   |       |   |   |-- ✅ isValidES3Identifier.js.map
|   |   |       |   |   |-- ✅ isValidIdentifier.js
|   |   |       |   |   |-- ✅ isValidIdentifier.js.map
|   |   |       |   |   |-- ✅ isVar.js
|   |   |       |   |   |-- ✅ isVar.js.map
|   |   |       |   |   |-- ✅ matchesPattern.js
|   |   |       |   |   |-- ✅ matchesPattern.js.map
|   |   |       |   |   |-- ✅ validate.js
|   |   |       |   |   \-- ✅ validate.js.map
|   |   |       |   |-- ✅ index-legacy.d.ts
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   |-- ✅ index.js
|   |   |       |   |-- ✅ index.js.flow
|   |   |       |   \-- ✅ index.js.map
|   |   |       |-- ✅ LICENSE
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @esbuild/
|   |   |   \-- ✅ win32-x64/
|   |   |       |-- ✅ esbuild.exe
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @isaacs/
|   |   |   \-- ✅ cliui/
|   |   |       |-- ✅ build/
|   |   |       |   |-- ✅ lib/
|   |   |       |   |   \-- ✅ index.js
|   |   |       |   |-- ✅ index.cjs
|   |   |       |   \-- ✅ index.d.cts
|   |   |       |-- ✅ index.mjs
|   |   |       |-- ✅ LICENSE.txt
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @jest/
|   |   |   |-- ✅ expect-utils/
|   |   |   |   |-- ✅ build/
|   |   |   |   |   |-- ✅ immutableUtils.js
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ jasmineUtils.js
|   |   |   |   |   |-- ✅ types.js
|   |   |   |   |   \-- ✅ utils.js
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ schemas/
|   |   |   |   |-- ✅ build/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   \-- ✅ types/
|   |   |       |-- ✅ build/
|   |   |       |   |-- ✅ Circus.js
|   |   |       |   |-- ✅ Config.js
|   |   |       |   |-- ✅ Global.js
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   |-- ✅ index.js
|   |   |       |   |-- ✅ TestResult.js
|   |   |       |   \-- ✅ Transform.js
|   |   |       |-- ✅ LICENSE
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @jridgewell/
|   |   |   |-- ✅ gen-mapping/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ types/
|   |   |   |   |   |   |-- ✅ gen-mapping.d.ts
|   |   |   |   |   |   |-- ✅ set-array.d.ts
|   |   |   |   |   |   |-- ✅ sourcemap-segment.d.ts
|   |   |   |   |   |   \-- ✅ types.d.ts
|   |   |   |   |   |-- ✅ gen-mapping.mjs
|   |   |   |   |   |-- ✅ gen-mapping.mjs.map
|   |   |   |   |   |-- ✅ gen-mapping.umd.js
|   |   |   |   |   \-- ✅ gen-mapping.umd.js.map
|   |   |   |   |-- ✅ src/
|   |   |   |   |   |-- ✅ gen-mapping.ts
|   |   |   |   |   |-- ✅ set-array.ts
|   |   |   |   |   |-- ✅ sourcemap-segment.ts
|   |   |   |   |   \-- ✅ types.ts
|   |   |   |   |-- ✅ types/
|   |   |   |   |   |-- ✅ gen-mapping.d.cts
|   |   |   |   |   |-- ✅ gen-mapping.d.cts.map
|   |   |   |   |   |-- ✅ gen-mapping.d.mts
|   |   |   |   |   |-- ✅ gen-mapping.d.mts.map
|   |   |   |   |   |-- ✅ set-array.d.cts
|   |   |   |   |   |-- ✅ set-array.d.cts.map
|   |   |   |   |   |-- ✅ set-array.d.mts
|   |   |   |   |   |-- ✅ set-array.d.mts.map
|   |   |   |   |   |-- ✅ sourcemap-segment.d.cts
|   |   |   |   |   |-- ✅ sourcemap-segment.d.cts.map
|   |   |   |   |   |-- ✅ sourcemap-segment.d.mts
|   |   |   |   |   |-- ✅ sourcemap-segment.d.mts.map
|   |   |   |   |   |-- ✅ types.d.cts
|   |   |   |   |   |-- ✅ types.d.cts.map
|   |   |   |   |   |-- ✅ types.d.mts
|   |   |   |   |   \-- ✅ types.d.mts.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ remapping/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ remapping.mjs
|   |   |   |   |   |-- ✅ remapping.mjs.map
|   |   |   |   |   |-- ✅ remapping.umd.js
|   |   |   |   |   \-- ✅ remapping.umd.js.map
|   |   |   |   |-- ✅ src/
|   |   |   |   |   |-- ✅ build-source-map-tree.ts
|   |   |   |   |   |-- ✅ remapping.ts
|   |   |   |   |   |-- ✅ source-map-tree.ts
|   |   |   |   |   |-- ✅ source-map.ts
|   |   |   |   |   \-- ✅ types.ts
|   |   |   |   |-- ✅ types/
|   |   |   |   |   |-- ✅ build-source-map-tree.d.cts
|   |   |   |   |   |-- ✅ build-source-map-tree.d.cts.map
|   |   |   |   |   |-- ✅ build-source-map-tree.d.mts
|   |   |   |   |   |-- ✅ build-source-map-tree.d.mts.map
|   |   |   |   |   |-- ✅ remapping.d.cts
|   |   |   |   |   |-- ✅ remapping.d.cts.map
|   |   |   |   |   |-- ✅ remapping.d.mts
|   |   |   |   |   |-- ✅ remapping.d.mts.map
|   |   |   |   |   |-- ✅ source-map-tree.d.cts
|   |   |   |   |   |-- ✅ source-map-tree.d.cts.map
|   |   |   |   |   |-- ✅ source-map-tree.d.mts
|   |   |   |   |   |-- ✅ source-map-tree.d.mts.map
|   |   |   |   |   |-- ✅ source-map.d.cts
|   |   |   |   |   |-- ✅ source-map.d.cts.map
|   |   |   |   |   |-- ✅ source-map.d.mts
|   |   |   |   |   |-- ✅ source-map.d.mts.map
|   |   |   |   |   |-- ✅ types.d.cts
|   |   |   |   |   |-- ✅ types.d.cts.map
|   |   |   |   |   |-- ✅ types.d.mts
|   |   |   |   |   \-- ✅ types.d.mts.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ resolve-uri/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ types/
|   |   |   |   |   |   \-- ✅ resolve-uri.d.ts
|   |   |   |   |   |-- ✅ resolve-uri.mjs
|   |   |   |   |   |-- ✅ resolve-uri.mjs.map
|   |   |   |   |   |-- ✅ resolve-uri.umd.js
|   |   |   |   |   \-- ✅ resolve-uri.umd.js.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ sourcemap-codec/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ sourcemap-codec.mjs
|   |   |   |   |   |-- ✅ sourcemap-codec.mjs.map
|   |   |   |   |   |-- ✅ sourcemap-codec.umd.js
|   |   |   |   |   \-- ✅ sourcemap-codec.umd.js.map
|   |   |   |   |-- ✅ src/
|   |   |   |   |   |-- ✅ scopes.ts
|   |   |   |   |   |-- ✅ sourcemap-codec.ts
|   |   |   |   |   |-- ✅ strings.ts
|   |   |   |   |   \-- ✅ vlq.ts
|   |   |   |   |-- ✅ types/
|   |   |   |   |   |-- ✅ scopes.d.cts
|   |   |   |   |   |-- ✅ scopes.d.cts.map
|   |   |   |   |   |-- ✅ scopes.d.mts
|   |   |   |   |   |-- ✅ scopes.d.mts.map
|   |   |   |   |   |-- ✅ sourcemap-codec.d.cts
|   |   |   |   |   |-- ✅ sourcemap-codec.d.cts.map
|   |   |   |   |   |-- ✅ sourcemap-codec.d.mts
|   |   |   |   |   |-- ✅ sourcemap-codec.d.mts.map
|   |   |   |   |   |-- ✅ strings.d.cts
|   |   |   |   |   |-- ✅ strings.d.cts.map
|   |   |   |   |   |-- ✅ strings.d.mts
|   |   |   |   |   |-- ✅ strings.d.mts.map
|   |   |   |   |   |-- ✅ vlq.d.cts
|   |   |   |   |   |-- ✅ vlq.d.cts.map
|   |   |   |   |   |-- ✅ vlq.d.mts
|   |   |   |   |   \-- ✅ vlq.d.mts.map
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   \-- ✅ trace-mapping/
|   |   |       |-- ✅ dist/
|   |   |       |   |-- ✅ trace-mapping.mjs
|   |   |       |   |-- ✅ trace-mapping.mjs.map
|   |   |       |   |-- ✅ trace-mapping.umd.js
|   |   |       |   \-- ✅ trace-mapping.umd.js.map
|   |   |       |-- ✅ src/
|   |   |       |   |-- ✅ binary-search.ts
|   |   |       |   |-- ✅ by-source.ts
|   |   |       |   |-- ✅ flatten-map.ts
|   |   |       |   |-- ✅ resolve.ts
|   |   |       |   |-- ✅ sort.ts
|   |   |       |   |-- ✅ sourcemap-segment.ts
|   |   |       |   |-- ✅ strip-filename.ts
|   |   |       |   |-- ✅ trace-mapping.ts
|   |   |       |   \-- ✅ types.ts
|   |   |       |-- ✅ types/
|   |   |       |   |-- ✅ binary-search.d.cts
|   |   |       |   |-- ✅ binary-search.d.cts.map
|   |   |       |   |-- ✅ binary-search.d.mts
|   |   |       |   |-- ✅ binary-search.d.mts.map
|   |   |       |   |-- ✅ by-source.d.cts
|   |   |       |   |-- ✅ by-source.d.cts.map
|   |   |       |   |-- ✅ by-source.d.mts
|   |   |       |   |-- ✅ by-source.d.mts.map
|   |   |       |   |-- ✅ flatten-map.d.cts
|   |   |       |   |-- ✅ flatten-map.d.cts.map
|   |   |       |   |-- ✅ flatten-map.d.mts
|   |   |       |   |-- ✅ flatten-map.d.mts.map
|   |   |       |   |-- ✅ resolve.d.cts
|   |   |       |   |-- ✅ resolve.d.cts.map
|   |   |       |   |-- ✅ resolve.d.mts
|   |   |       |   |-- ✅ resolve.d.mts.map
|   |   |       |   |-- ✅ sort.d.cts
|   |   |       |   |-- ✅ sort.d.cts.map
|   |   |       |   |-- ✅ sort.d.mts
|   |   |       |   |-- ✅ sort.d.mts.map
|   |   |       |   |-- ✅ sourcemap-segment.d.cts
|   |   |       |   |-- ✅ sourcemap-segment.d.cts.map
|   |   |       |   |-- ✅ sourcemap-segment.d.mts
|   |   |       |   |-- ✅ sourcemap-segment.d.mts.map
|   |   |       |   |-- ✅ strip-filename.d.cts
|   |   |       |   |-- ✅ strip-filename.d.cts.map
|   |   |       |   |-- ✅ strip-filename.d.mts
|   |   |       |   |-- ✅ strip-filename.d.mts.map
|   |   |       |   |-- ✅ trace-mapping.d.cts
|   |   |       |   |-- ✅ trace-mapping.d.cts.map
|   |   |       |   |-- ✅ trace-mapping.d.mts
|   |   |       |   |-- ✅ trace-mapping.d.mts.map
|   |   |       |   |-- ✅ types.d.cts
|   |   |       |   |-- ✅ types.d.cts.map
|   |   |       |   |-- ✅ types.d.mts
|   |   |       |   \-- ✅ types.d.mts.map
|   |   |       |-- ✅ LICENSE
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @kurkle/
|   |   |   \-- ✅ color/
|   |   |       |-- ✅ dist/
|   |   |       |   |-- ✅ color.cjs
|   |   |       |   |-- ✅ color.d.ts
|   |   |       |   |-- ✅ color.esm.js
|   |   |       |   |-- ✅ color.min.js
|   |   |       |   \-- ✅ color.min.js.map
|   |   |       |-- ✅ LICENSE.md
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @nodelib/
|   |   |   |-- ✅ fs.scandir/
|   |   |   |   |-- ✅ out/
|   |   |   |   |   |-- ✅ adapters/
|   |   |   |   |   |   |-- ✅ fs.d.ts
|   |   |   |   |   |   \-- ✅ fs.js
|   |   |   |   |   |-- ✅ providers/
|   |   |   |   |   |   |-- ✅ async.d.ts
|   |   |   |   |   |   |-- ✅ async.js
|   |   |   |   |   |   |-- ✅ common.d.ts
|   |   |   |   |   |   |-- ✅ common.js
|   |   |   |   |   |   |-- ✅ sync.d.ts
|   |   |   |   |   |   \-- ✅ sync.js
|   |   |   |   |   |-- ✅ types/
|   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   |-- ✅ utils/
|   |   |   |   |   |   |-- ✅ fs.d.ts
|   |   |   |   |   |   |-- ✅ fs.js
|   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   |-- ✅ constants.d.ts
|   |   |   |   |   |-- ✅ constants.js
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ settings.d.ts
|   |   |   |   |   \-- ✅ settings.js
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ fs.stat/
|   |   |   |   |-- ✅ out/
|   |   |   |   |   |-- ✅ adapters/
|   |   |   |   |   |   |-- ✅ fs.d.ts
|   |   |   |   |   |   \-- ✅ fs.js
|   |   |   |   |   |-- ✅ providers/
|   |   |   |   |   |   |-- ✅ async.d.ts
|   |   |   |   |   |   |-- ✅ async.js
|   |   |   |   |   |   |-- ✅ sync.d.ts
|   |   |   |   |   |   \-- ✅ sync.js
|   |   |   |   |   |-- ✅ types/
|   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ settings.d.ts
|   |   |   |   |   \-- ✅ settings.js
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   \-- ✅ fs.walk/
|   |   |       |-- ✅ out/
|   |   |       |   |-- ✅ providers/
|   |   |       |   |   |-- ✅ async.d.ts
|   |   |       |   |   |-- ✅ async.js
|   |   |       |   |   |-- ✅ index.d.ts
|   |   |       |   |   |-- ✅ index.js
|   |   |       |   |   |-- ✅ stream.d.ts
|   |   |       |   |   |-- ✅ stream.js
|   |   |       |   |   |-- ✅ sync.d.ts
|   |   |       |   |   \-- ✅ sync.js
|   |   |       |   |-- ✅ readers/
|   |   |       |   |   |-- ✅ async.d.ts
|   |   |       |   |   |-- ✅ async.js
|   |   |       |   |   |-- ✅ common.d.ts
|   |   |       |   |   |-- ✅ common.js
|   |   |       |   |   |-- ✅ reader.d.ts
|   |   |       |   |   |-- ✅ reader.js
|   |   |       |   |   |-- ✅ sync.d.ts
|   |   |       |   |   \-- ✅ sync.js
|   |   |       |   |-- ✅ types/
|   |   |       |   |   |-- ✅ index.d.ts
|   |   |       |   |   \-- ✅ index.js
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   |-- ✅ index.js
|   |   |       |   |-- ✅ settings.d.ts
|   |   |       |   \-- ✅ settings.js
|   |   |       |-- ✅ LICENSE
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @pkgjs/
|   |   |   \-- ✅ parseargs/
|   |   |       |-- ✅ examples/
|   |   |       |   |-- ✅ is-default-value.js
|   |   |       |   |-- ✅ limit-long-syntax.js
|   |   |       |   |-- ✅ negate.js
|   |   |       |   |-- ✅ no-repeated-options.js
|   |   |       |   |-- ✅ ordered-options.mjs
|   |   |       |   \-- ✅ simple-hard-coded.js
|   |   |       |-- ✅ internal/
|   |   |       |   |-- ✅ errors.js
|   |   |       |   |-- ✅ primordials.js
|   |   |       |   |-- ✅ util.js
|   |   |       |   \-- ✅ validators.js
|   |   |       |-- ✅ .editorconfig
|   |   |       |-- ✅ CHANGELOG.md
|   |   |       |-- ✅ index.js
|   |   |       |-- ✅ LICENSE
|   |   |       |-- ✅ package.json
|   |   |       |-- ✅ README.md
|   |   |       \-- ✅ utils.js
|   |   |-- ✅ @rolldown/
|   |   |   \-- ✅ pluginutils/
|   |   |       |-- ✅ dist/
|   |   |       |   |-- ✅ index.cjs
|   |   |       |   |-- ✅ index.d.cts
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   \-- ✅ index.js
|   |   |       |-- ✅ LICENSE
|   |   |       \-- ✅ package.json
|   |   |-- ✅ @rollup/
|   |   |   |-- ✅ rollup-win32-x64-gnu/
|   |   |   |   |-- ✅ package.json
|   |   |   |   |-- ✅ README.md
|   |   |   |   \-- ✅ rollup.win32-x64-gnu.node
|   |   |   \-- ✅ rollup-win32-x64-msvc/
|   |   |       |-- ✅ package.json
|   |   |       |-- ✅ README.md
|   |   |       \-- ✅ rollup.win32-x64-msvc.node
|   |   |-- ✅ @sinclair/
|   |   |   \-- ✅ typebox/
|   |   |       |-- ✅ compiler/
|   |   |       |   |-- ✅ compiler.d.ts
|   |   |       |   |-- ✅ compiler.js
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   \-- ✅ index.js
|   |   |       |-- ✅ errors/
|   |   |       |   |-- ✅ errors.d.ts
|   |   |       |   |-- ✅ errors.js
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   \-- ✅ index.js
|   |   |       |-- ✅ system/
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   |-- ✅ index.js
|   |   |       |   |-- ✅ system.d.ts
|   |   |       |   \-- ✅ system.js
|   |   |       |-- ✅ value/
|   |   |       |   |-- ✅ cast.d.ts
|   |   |       |   |-- ✅ cast.js
|   |   |       |   |-- ✅ check.d.ts
|   |   |       |   |-- ✅ check.js
|   |   |       |   |-- ✅ clone.d.ts
|   |   |       |   |-- ✅ clone.js
|   |   |       |   |-- ✅ convert.d.ts
|   |   |       |   |-- ✅ convert.js
|   |   |       |   |-- ✅ create.d.ts
|   |   |       |   |-- ✅ create.js
|   |   |       |   |-- ✅ delta.d.ts
|   |   |       |   |-- ✅ delta.js
|   |   |       |   |-- ✅ equal.d.ts
|   |   |       |   |-- ✅ equal.js
|   |   |       |   |-- ✅ hash.d.ts
|   |   |       |   |-- ✅ hash.js
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   |-- ✅ index.js
|   |   |       |   |-- ✅ is.d.ts
|   |   |       |   |-- ✅ is.js
|   |   |       |   |-- ✅ mutate.d.ts
|   |   |       |   |-- ✅ mutate.js
|   |   |       |   |-- ✅ pointer.d.ts
|   |   |       |   |-- ✅ pointer.js
|   |   |       |   |-- ✅ value.d.ts
|   |   |       |   \-- ✅ value.js
|   |   |       |-- ✅ license
|   |   |       |-- ✅ package.json
|   |   |       |-- ✅ readme.md
|   |   |       |-- ✅ typebox.d.ts
|   |   |       \-- ✅ typebox.js
|   |   |-- ✅ @testing-library/
|   |   |   |-- ✅ dom/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ @testing-library/
|   |   |   |   |   |   |-- ✅ dom.cjs.js
|   |   |   |   |   |   |-- ✅ dom.esm.js
|   |   |   |   |   |   |-- ✅ dom.umd.js
|   |   |   |   |   |   |-- ✅ dom.umd.js.map
|   |   |   |   |   |   |-- ✅ dom.umd.min.js
|   |   |   |   |   |   \-- ✅ dom.umd.min.js.map
|   |   |   |   |   |-- ✅ queries/
|   |   |   |   |   |   |-- ✅ all-utils.js
|   |   |   |   |   |   |-- ✅ alt-text.js
|   |   |   |   |   |   |-- ✅ display-value.js
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ label-text.js
|   |   |   |   |   |   |-- ✅ placeholder-text.js
|   |   |   |   |   |   |-- ✅ role.js
|   |   |   |   |   |   |-- ✅ test-id.js
|   |   |   |   |   |   |-- ✅ text.js
|   |   |   |   |   |   \-- ✅ title.js
|   |   |   |   |   |-- ✅ config.js
|   |   |   |   |   |-- ✅ DOMElementFilter.js
|   |   |   |   |   |-- ✅ event-map.js
|   |   |   |   |   |-- ✅ events.js
|   |   |   |   |   |-- ✅ get-node-text.js
|   |   |   |   |   |-- ✅ get-queries-for-element.js
|   |   |   |   |   |-- ✅ get-user-code-frame.js
|   |   |   |   |   |-- ✅ helpers.js
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ label-helpers.js
|   |   |   |   |   |-- ✅ matches.js
|   |   |   |   |   |-- ✅ pretty-dom.js
|   |   |   |   |   |-- ✅ query-helpers.js
|   |   |   |   |   |-- ✅ role-helpers.js
|   |   |   |   |   |-- ✅ screen.js
|   |   |   |   |   |-- ✅ suggestions.js
|   |   |   |   |   |-- ✅ wait-for-element-to-be-removed.js
|   |   |   |   |   \-- ✅ wait-for.js
|   |   |   |   |-- ✅ node_modules/
|   |   |   |   |   |-- ✅ aria-query/
|   |   |   |   |   |   |-- ✅ lib/
|   |   |   |   |   |   |   |-- ✅ etc/
|   |   |   |   |   |   |   |   \-- ✅ roles/
|   |   |   |   |   |   |   |       |-- ✅ abstract/
|   |   |   |   |   |   |   |       |   |-- ✅ commandRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ compositeRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ inputRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ landmarkRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ rangeRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ roletypeRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ sectionheadRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ sectionRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ selectRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ structureRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ widgetRole.js
|   |   |   |   |   |   |   |       |   \-- ✅ windowRole.js
|   |   |   |   |   |   |   |       |-- ✅ dpub/
|   |   |   |   |   |   |   |       |   |-- ✅ docAbstractRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docAcknowledgmentsRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docAfterwordRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docAppendixRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docBacklinkRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docBiblioentryRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docBibliographyRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docBibliorefRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docChapterRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docColophonRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docConclusionRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docCoverRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docCreditRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docCreditsRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docDedicationRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docEndnoteRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docEndnotesRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docEpigraphRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docEpilogueRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docErrataRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docExampleRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docFootnoteRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docForewordRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docGlossaryRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docGlossrefRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docIndexRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docIntroductionRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docNoterefRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docNoticeRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docPagebreakRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docPagelistRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docPartRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docPrefaceRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docPrologueRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docPullquoteRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docQnaRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docSubtitleRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ docTipRole.js
|   |   |   |   |   |   |   |       |   \-- ✅ docTocRole.js
|   |   |   |   |   |   |   |       |-- ✅ graphics/
|   |   |   |   |   |   |   |       |   |-- ✅ graphicsDocumentRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ graphicsObjectRole.js
|   |   |   |   |   |   |   |       |   \-- ✅ graphicsSymbolRole.js
|   |   |   |   |   |   |   |       |-- ✅ literal/
|   |   |   |   |   |   |   |       |   |-- ✅ alertdialogRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ alertRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ applicationRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ articleRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ bannerRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ blockquoteRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ buttonRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ captionRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ cellRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ checkboxRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ codeRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ columnheaderRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ comboboxRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ complementaryRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ contentinfoRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ definitionRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ deletionRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ dialogRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ directoryRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ documentRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ emphasisRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ feedRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ figureRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ formRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ genericRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ graphicsDocumentRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ graphicsObjectRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ graphicsSymbolRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ gridcellRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ gridRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ groupRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ headingRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ imgRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ insertionRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ linkRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ listboxRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ listitemRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ listRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ logRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ mainRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ marqueeRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ mathRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ menubarRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ menuitemcheckboxRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ menuitemradioRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ menuitemRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ menuRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ meterRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ navigationRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ noneRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ noteRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ optionRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ paragraphRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ presentationRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ progressbarRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ radiogroupRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ radioRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ regionRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ rowgroupRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ rowheaderRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ rowRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ scrollbarRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ searchboxRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ searchRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ separatorRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ sliderRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ spinbuttonRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ statusRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ strongRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ subscriptRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ superscriptRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ switchRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ tableRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ tablistRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ tabpanelRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ tabRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ termRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ textboxRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ timeRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ timerRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ toolbarRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ tooltipRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ treegridRole.js
|   |   |   |   |   |   |   |       |   |-- ✅ treeitemRole.js
|   |   |   |   |   |   |   |       |   \-- ✅ treeRole.js
|   |   |   |   |   |   |   |       |-- ✅ ariaAbstractRoles.js
|   |   |   |   |   |   |   |       |-- ✅ ariaDpubRoles.js
|   |   |   |   |   |   |   |       |-- ✅ ariaGraphicsRoles.js
|   |   |   |   |   |   |   |       \-- ✅ ariaLiteralRoles.js
|   |   |   |   |   |   |   |-- ✅ util/
|   |   |   |   |   |   |   |   |-- ✅ iterationDecorator.js
|   |   |   |   |   |   |   |   \-- ✅ iteratorProxy.js
|   |   |   |   |   |   |   |-- ✅ ariaPropsMap.js
|   |   |   |   |   |   |   |-- ✅ domMap.js
|   |   |   |   |   |   |   |-- ✅ elementRoleMap.js
|   |   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |   |-- ✅ roleElementMap.js
|   |   |   |   |   |   |   \-- ✅ rolesMap.js
|   |   |   |   |   |   |-- ✅ LICENSE
|   |   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |   \-- ✅ README.md
|   |   |   |   |   \-- ✅ dom-accessibility-api/
|   |   |   |   |       |-- ✅ dist/
|   |   |   |   |       |   |-- ✅ polyfills/
|   |   |   |   |       |   |   |-- ✅ array.from.d.ts
|   |   |   |   |       |   |   |-- ✅ array.from.d.ts.map
|   |   |   |   |       |   |   |-- ✅ array.from.js
|   |   |   |   |       |   |   |-- ✅ array.from.js.map
|   |   |   |   |       |   |   |-- ✅ array.from.mjs
|   |   |   |   |       |   |   |-- ✅ array.from.mjs.map
|   |   |   |   |       |   |   |-- ✅ iterator.d.js
|   |   |   |   |       |   |   |-- ✅ iterator.d.js.map
|   |   |   |   |       |   |   |-- ✅ iterator.d.mjs
|   |   |   |   |       |   |   |-- ✅ iterator.d.mjs.map
|   |   |   |   |       |   |   |-- ✅ SetLike.d.ts
|   |   |   |   |       |   |   |-- ✅ SetLike.d.ts.map
|   |   |   |   |       |   |   |-- ✅ SetLike.js
|   |   |   |   |       |   |   |-- ✅ SetLike.js.map
|   |   |   |   |       |   |   |-- ✅ SetLike.mjs
|   |   |   |   |       |   |   \-- ✅ SetLike.mjs.map
|   |   |   |   |       |   |-- ✅ accessible-description.d.ts
|   |   |   |   |       |   |-- ✅ accessible-description.d.ts.map
|   |   |   |   |       |   |-- ✅ accessible-description.js
|   |   |   |   |       |   |-- ✅ accessible-description.js.map
|   |   |   |   |       |   |-- ✅ accessible-description.mjs
|   |   |   |   |       |   |-- ✅ accessible-description.mjs.map
|   |   |   |   |       |   |-- ✅ accessible-name-and-description.d.ts
|   |   |   |   |       |   |-- ✅ accessible-name-and-description.d.ts.map
|   |   |   |   |       |   |-- ✅ accessible-name-and-description.js
|   |   |   |   |       |   |-- ✅ accessible-name-and-description.js.map
|   |   |   |   |       |   |-- ✅ accessible-name-and-description.mjs
|   |   |   |   |       |   |-- ✅ accessible-name-and-description.mjs.map
|   |   |   |   |       |   |-- ✅ accessible-name.d.ts
|   |   |   |   |       |   |-- ✅ accessible-name.d.ts.map
|   |   |   |   |       |   |-- ✅ accessible-name.js
|   |   |   |   |       |   |-- ✅ accessible-name.js.map
|   |   |   |   |       |   |-- ✅ accessible-name.mjs
|   |   |   |   |       |   |-- ✅ accessible-name.mjs.map
|   |   |   |   |       |   |-- ✅ getRole.d.ts
|   |   |   |   |       |   |-- ✅ getRole.d.ts.map
|   |   |   |   |       |   |-- ✅ getRole.js
|   |   |   |   |       |   |-- ✅ getRole.js.map
|   |   |   |   |       |   |-- ✅ getRole.mjs
|   |   |   |   |       |   |-- ✅ getRole.mjs.map
|   |   |   |   |       |   |-- ✅ index.d.ts
|   |   |   |   |       |   |-- ✅ index.d.ts.map
|   |   |   |   |       |   |-- ✅ index.js
|   |   |   |   |       |   |-- ✅ index.js.map
|   |   |   |   |       |   |-- ✅ index.mjs
|   |   |   |   |       |   |-- ✅ index.mjs.map
|   |   |   |   |       |   |-- ✅ is-inaccessible.d.ts
|   |   |   |   |       |   |-- ✅ is-inaccessible.d.ts.map
|   |   |   |   |       |   |-- ✅ is-inaccessible.js
|   |   |   |   |       |   |-- ✅ is-inaccessible.js.map
|   |   |   |   |       |   |-- ✅ is-inaccessible.mjs
|   |   |   |   |       |   |-- ✅ is-inaccessible.mjs.map
|   |   |   |   |       |   |-- ✅ util.d.ts
|   |   |   |   |       |   |-- ✅ util.d.ts.map
|   |   |   |   |       |   |-- ✅ util.js
|   |   |   |   |       |   |-- ✅ util.js.map
|   |   |   |   |       |   |-- ✅ util.mjs
|   |   |   |   |       |   \-- ✅ util.mjs.map
|   |   |   |   |       |-- ✅ .browserslistrc
|   |   |   |   |       |-- ✅ LICENSE.md
|   |   |   |   |       |-- ✅ package.json
|   |   |   |   |       \-- ✅ README.md
|   |   |   |   |-- ✅ types/
|   |   |   |   |   |-- ✅ config.d.ts
|   |   |   |   |   |-- ✅ events.d.ts
|   |   |   |   |   |-- ✅ get-node-text.d.ts
|   |   |   |   |   |-- ✅ get-queries-for-element.d.ts
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ matches.d.ts
|   |   |   |   |   |-- ✅ pretty-dom.d.ts
|   |   |   |   |   |-- ✅ queries.d.ts
|   |   |   |   |   |-- ✅ query-helpers.d.ts
|   |   |   |   |   |-- ✅ role-helpers.d.ts
|   |   |   |   |   |-- ✅ screen.d.ts
|   |   |   |   |   |-- ✅ suggestions.d.ts
|   |   |   |   |   |-- ✅ wait-for-element-to-be-removed.d.ts
|   |   |   |   |   \-- ✅ wait-for.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ jest-dom/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.mjs
|   |   |   |   |   |-- ✅ jest-globals.js
|   |   |   |   |   |-- ✅ jest-globals.mjs
|   |   |   |   |   |-- ✅ matchers-8f81fc78.mjs
|   |   |   |   |   |-- ✅ matchers-a1259dd2.js
|   |   |   |   |   |-- ✅ matchers.js
|   |   |   |   |   |-- ✅ matchers.mjs
|   |   |   |   |   |-- ✅ vitest.js
|   |   |   |   |   \-- ✅ vitest.mjs
|   |   |   |   |-- ✅ types/
|   |   |   |   |   |-- ✅ __tests__/
|   |   |   |   |   |   |-- ✅ bun/
|   |   |   |   |   |   |   |-- ✅ bun-custom-expect-types.test.ts
|   |   |   |   |   |   |   |-- ✅ bun-types.test.ts
|   |   |   |   |   |   |   \-- ✅ tsconfig.json
|   |   |   |   |   |   |-- ✅ jest/
|   |   |   |   |   |   |   |-- ✅ jest-custom-expect-types.test.ts
|   |   |   |   |   |   |   |-- ✅ jest-types.test.ts
|   |   |   |   |   |   |   \-- ✅ tsconfig.json
|   |   |   |   |   |   |-- ✅ jest-globals/
|   |   |   |   |   |   |   |-- ✅ jest-globals-custom-expect-types.test.ts
|   |   |   |   |   |   |   |-- ✅ jest-globals-types.test.ts
|   |   |   |   |   |   |   \-- ✅ tsconfig.json
|   |   |   |   |   |   \-- ✅ vitest/
|   |   |   |   |   |       |-- ✅ tsconfig.json
|   |   |   |   |   |       |-- ✅ vitest-custom-expect-types.test.ts
|   |   |   |   |   |       \-- ✅ vitest-types.test.ts
|   |   |   |   |   |-- ✅ bun.d.ts
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ jest-globals.d.ts
|   |   |   |   |   |-- ✅ jest.d.ts
|   |   |   |   |   |-- ✅ matchers-standalone.d.ts
|   |   |   |   |   |-- ✅ matchers.d.ts
|   |   |   |   |   \-- ✅ vitest.d.ts
|   |   |   |   |-- ✅ CHANGELOG.md
|   |   |   |   |-- ✅ jest-globals.d.ts
|   |   |   |   |-- ✅ jest-globals.js
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ matchers.d.ts
|   |   |   |   |-- ✅ matchers.js
|   |   |   |   |-- ✅ package.json
|   |   |   |   |-- ✅ README.md
|   |   |   |   |-- ✅ vitest.d.ts
|   |   |   |   \-- ✅ vitest.js
|   |   |   \-- ✅ react/
|   |   |       |-- ✅ dist/
|   |   |       |   |-- ✅ @testing-library/
|   |   |       |   |   |-- ✅ react.cjs.js
|   |   |       |   |   |-- ✅ react.esm.js
|   |   |       |   |   |-- ✅ react.pure.cjs.js
|   |   |       |   |   |-- ✅ react.pure.esm.js
|   |   |       |   |   |-- ✅ react.pure.umd.js
|   |   |       |   |   |-- ✅ react.pure.umd.js.map
|   |   |       |   |   |-- ✅ react.pure.umd.min.js
|   |   |       |   |   |-- ✅ react.pure.umd.min.js.map
|   |   |       |   |   |-- ✅ react.umd.js
|   |   |       |   |   |-- ✅ react.umd.js.map
|   |   |       |   |   |-- ✅ react.umd.min.js
|   |   |       |   |   \-- ✅ react.umd.min.js.map
|   |   |       |   |-- ✅ act-compat.js
|   |   |       |   |-- ✅ config.js
|   |   |       |   |-- ✅ fire-event.js
|   |   |       |   |-- ✅ index.js
|   |   |       |   \-- ✅ pure.js
|   |   |       |-- ✅ types/
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   \-- ✅ pure.d.ts
|   |   |       |-- ✅ CHANGELOG.md
|   |   |       |-- ✅ dont-cleanup-after-each.js
|   |   |       |-- ✅ LICENSE
|   |   |       |-- ✅ package.json
|   |   |       |-- ✅ pure.d.ts
|   |   |       |-- ✅ pure.js
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @tootallnate/
|   |   |   \-- ✅ once/
|   |   |       |-- ✅ dist/
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   |-- ✅ index.js
|   |   |       |   |-- ✅ index.js.map
|   |   |       |   |-- ✅ overloaded-parameters.d.ts
|   |   |       |   |-- ✅ overloaded-parameters.js
|   |   |       |   |-- ✅ overloaded-parameters.js.map
|   |   |       |   |-- ✅ types.d.ts
|   |   |       |   |-- ✅ types.js
|   |   |       |   \-- ✅ types.js.map
|   |   |       |-- ✅ LICENSE
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @types/
|   |   |   |-- ✅ aria-query/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ babel__core/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ babel__generator/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ babel__template/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ babel__traverse/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ chai/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   |-- ✅ README.md
|   |   |   |   \-- ✅ register-should.d.ts
|   |   |   |-- ✅ d3-path/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ d3-shape/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ deep-eql/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ estree/
|   |   |   |   |-- ✅ flow.d.ts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ istanbul-lib-coverage/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ istanbul-lib-report/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ istanbul-reports/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ jest/
|   |   |   |   |-- ✅ node_modules/
|   |   |   |   |   |-- ✅ ansi-styles/
|   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ license
|   |   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |   \-- ✅ readme.md
|   |   |   |   |   |-- ✅ pretty-format/
|   |   |   |   |   |   |-- ✅ build/
|   |   |   |   |   |   |   |-- ✅ plugins/
|   |   |   |   |   |   |   |   |-- ✅ lib/
|   |   |   |   |   |   |   |   |   |-- ✅ escapeHTML.js
|   |   |   |   |   |   |   |   |   \-- ✅ markup.js
|   |   |   |   |   |   |   |   |-- ✅ AsymmetricMatcher.js
|   |   |   |   |   |   |   |   |-- ✅ DOMCollection.js
|   |   |   |   |   |   |   |   |-- ✅ DOMElement.js
|   |   |   |   |   |   |   |   |-- ✅ Immutable.js
|   |   |   |   |   |   |   |   |-- ✅ ReactElement.js
|   |   |   |   |   |   |   |   \-- ✅ ReactTestComponent.js
|   |   |   |   |   |   |   |-- ✅ collections.js
|   |   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |   \-- ✅ types.js
|   |   |   |   |   |   |-- ✅ LICENSE
|   |   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |   \-- ✅ README.md
|   |   |   |   |   \-- ✅ react-is/
|   |   |   |   |       |-- ✅ cjs/
|   |   |   |   |       |   |-- ✅ react-is.development.js
|   |   |   |   |       |   \-- ✅ react-is.production.min.js
|   |   |   |   |       |-- ✅ umd/
|   |   |   |   |       |   |-- ✅ react-is.development.js
|   |   |   |   |       |   \-- ✅ react-is.production.min.js
|   |   |   |   |       |-- ✅ index.js
|   |   |   |   |       |-- ✅ LICENSE
|   |   |   |   |       |-- ✅ package.json
|   |   |   |   |       \-- ✅ README.md
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ jsdom/
|   |   |   |   |-- ✅ base.d.ts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ moment/
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ node/
|   |   |   |   |-- ✅ assert/
|   |   |   |   |   \-- ✅ strict.d.ts
|   |   |   |   |-- ✅ compatibility/
|   |   |   |   |   |-- ✅ disposable.d.ts
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ indexable.d.ts
|   |   |   |   |   \-- ✅ iterators.d.ts
|   |   |   |   |-- ✅ dns/
|   |   |   |   |   \-- ✅ promises.d.ts
|   |   |   |   |-- ✅ fs/
|   |   |   |   |   \-- ✅ promises.d.ts
|   |   |   |   |-- ✅ readline/
|   |   |   |   |   \-- ✅ promises.d.ts
|   |   |   |   |-- ✅ stream/
|   |   |   |   |   |-- ✅ consumers.d.ts
|   |   |   |   |   |-- ✅ promises.d.ts
|   |   |   |   |   \-- ✅ web.d.ts
|   |   |   |   |-- ✅ timers/
|   |   |   |   |   \-- ✅ promises.d.ts
|   |   |   |   |-- ✅ ts5.6/
|   |   |   |   |   |-- ✅ buffer.buffer.d.ts
|   |   |   |   |   |-- ✅ globals.typedarray.d.ts
|   |   |   |   |   \-- ✅ index.d.ts
|   |   |   |   |-- ✅ web-globals/
|   |   |   |   |   |-- ✅ abortcontroller.d.ts
|   |   |   |   |   |-- ✅ domexception.d.ts
|   |   |   |   |   |-- ✅ events.d.ts
|   |   |   |   |   \-- ✅ fetch.d.ts
|   |   |   |   |-- ✅ assert.d.ts
|   |   |   |   |-- ✅ async_hooks.d.ts
|   |   |   |   |-- ✅ buffer.buffer.d.ts
|   |   |   |   |-- ✅ buffer.d.ts
|   |   |   |   |-- ✅ child_process.d.ts
|   |   |   |   |-- ✅ cluster.d.ts
|   |   |   |   |-- ✅ console.d.ts
|   |   |   |   |-- ✅ constants.d.ts
|   |   |   |   |-- ✅ crypto.d.ts
|   |   |   |   |-- ✅ dgram.d.ts
|   |   |   |   |-- ✅ diagnostics_channel.d.ts
|   |   |   |   |-- ✅ dns.d.ts
|   |   |   |   |-- ✅ domain.d.ts
|   |   |   |   |-- ✅ events.d.ts
|   |   |   |   |-- ✅ fs.d.ts
|   |   |   |   |-- ✅ globals.d.ts
|   |   |   |   |-- ✅ globals.typedarray.d.ts
|   |   |   |   |-- ✅ http.d.ts
|   |   |   |   |-- ✅ http2.d.ts
|   |   |   |   |-- ✅ https.d.ts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ inspector.generated.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ module.d.ts
|   |   |   |   |-- ✅ net.d.ts
|   |   |   |   |-- ✅ os.d.ts
|   |   |   |   |-- ✅ package.json
|   |   |   |   |-- ✅ path.d.ts
|   |   |   |   |-- ✅ perf_hooks.d.ts
|   |   |   |   |-- ✅ process.d.ts
|   |   |   |   |-- ✅ punycode.d.ts
|   |   |   |   |-- ✅ querystring.d.ts
|   |   |   |   |-- ✅ readline.d.ts
|   |   |   |   |-- ✅ README.md
|   |   |   |   |-- ✅ repl.d.ts
|   |   |   |   |-- ✅ sea.d.ts
|   |   |   |   |-- ✅ stream.d.ts
|   |   |   |   |-- ✅ string_decoder.d.ts
|   |   |   |   |-- ✅ test.d.ts
|   |   |   |   |-- ✅ timers.d.ts
|   |   |   |   |-- ✅ tls.d.ts
|   |   |   |   |-- ✅ trace_events.d.ts
|   |   |   |   |-- ✅ tty.d.ts
|   |   |   |   |-- ✅ url.d.ts
|   |   |   |   |-- ✅ util.d.ts
|   |   |   |   |-- ✅ v8.d.ts
|   |   |   |   |-- ✅ vm.d.ts
|   |   |   |   |-- ✅ wasi.d.ts
|   |   |   |   |-- ✅ worker_threads.d.ts
|   |   |   |   \-- ✅ zlib.d.ts
|   |   |   |-- ✅ prop-types/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ react/
|   |   |   |   |-- ✅ ts5.0/
|   |   |   |   |   |-- ✅ canary.d.ts
|   |   |   |   |   |-- ✅ experimental.d.ts
|   |   |   |   |   |-- ✅ global.d.ts
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ jsx-dev-runtime.d.ts
|   |   |   |   |   \-- ✅ jsx-runtime.d.ts
|   |   |   |   |-- ✅ canary.d.ts
|   |   |   |   |-- ✅ experimental.d.ts
|   |   |   |   |-- ✅ global.d.ts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ jsx-dev-runtime.d.ts
|   |   |   |   |-- ✅ jsx-runtime.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ react-dom/
|   |   |   |   |-- ✅ test-utils/
|   |   |   |   |   \-- ✅ index.d.ts
|   |   |   |   |-- ✅ canary.d.ts
|   |   |   |   |-- ✅ client.d.ts
|   |   |   |   |-- ✅ experimental.d.ts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   |-- ✅ README.md
|   |   |   |   \-- ✅ server.d.ts
|   |   |   |-- ✅ recharts/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ stack-utils/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ tough-cookie/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ yargs/
|   |   |   |   |-- ✅ helpers.d.mts
|   |   |   |   |-- ✅ helpers.d.ts
|   |   |   |   |-- ✅ index.d.mts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   |-- ✅ README.md
|   |   |   |   \-- ✅ yargs.d.ts
|   |   |   \-- ✅ yargs-parser/
|   |   |       |-- ✅ index.d.ts
|   |   |       |-- ✅ LICENSE
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @vitejs/
|   |   |   \-- ✅ plugin-react/
|   |   |       |-- ✅ dist/
|   |   |       |   |-- ✅ index.cjs
|   |   |       |   |-- ✅ index.d.cts
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   |-- ✅ index.js
|   |   |       |   \-- ✅ refresh-runtime.js
|   |   |       |-- ✅ LICENSE
|   |   |       |-- ✅ package.json
|   |   |       \-- ✅ README.md
|   |   |-- ✅ @vitest/
|   |   |   |-- ✅ expect/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ mocker/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ auto-register.d.ts
|   |   |   |   |   |-- ✅ auto-register.js
|   |   |   |   |   |-- ✅ browser.d.ts
|   |   |   |   |   |-- ✅ browser.js
|   |   |   |   |   |-- ✅ chunk-interceptor-native.js
|   |   |   |   |   |-- ✅ chunk-mocker.js
|   |   |   |   |   |-- ✅ chunk-pathe.M-eThtNZ.js
|   |   |   |   |   |-- ✅ chunk-registry.js
|   |   |   |   |   |-- ✅ chunk-utils.js
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ mocker.d-Ce9_ySj5.d.ts
|   |   |   |   |   |-- ✅ node.d.ts
|   |   |   |   |   |-- ✅ node.js
|   |   |   |   |   |-- ✅ redirect.d.ts
|   |   |   |   |   |-- ✅ redirect.js
|   |   |   |   |   |-- ✅ register.d.ts
|   |   |   |   |   |-- ✅ register.js
|   |   |   |   |   |-- ✅ registry.d-D765pazg.d.ts
|   |   |   |   |   \-- ✅ types.d-D_aRZRdy.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ pretty-format/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   \-- ✅ package.json
|   |   |   |-- ✅ runner/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ chunk-hooks.js
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ tasks.d-CkscK4of.d.ts
|   |   |   |   |   |-- ✅ types.d.ts
|   |   |   |   |   |-- ✅ types.js
|   |   |   |   |   |-- ✅ utils.d.ts
|   |   |   |   |   \-- ✅ utils.js
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   |-- ✅ README.md
|   |   |   |   |-- ✅ types.d.ts
|   |   |   |   \-- ✅ utils.d.ts
|   |   |   |-- ✅ snapshot/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ environment.d-DHdQ1Csl.d.ts
|   |   |   |   |   |-- ✅ environment.d.ts
|   |   |   |   |   |-- ✅ environment.js
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ manager.d.ts
|   |   |   |   |   |-- ✅ manager.js
|   |   |   |   |   \-- ✅ rawSnapshot.d-lFsMJFUd.d.ts
|   |   |   |   |-- ✅ environment.d.ts
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ manager.d.ts
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   |-- ✅ spy/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ LICENSE
|   |   |   |   |-- ✅ package.json
|   |   |   |   \-- ✅ README.md
|   |   |   \-- ✅ utils/
|   |   |       |-- ✅ dist/
|   |   |       |   |-- ✅ chunk-_commonjsHelpers.js
|   |   |       |   |-- ✅ diff.d.ts
|   |   |       |   |-- ✅ diff.js
|   |   |       |   |-- ✅ error.d.ts
|   |   |       |   |-- ✅ error.js
|   |   |       |   |-- ✅ helpers.d.ts
|   |   |       |   |-- ✅ helpers.js
|   |   |       |   |-- ✅ index.d.ts
|   |   |       |   |-- ✅ index.js
|   |   |       |   |-- ✅ source-map.d.ts
|   |   |       |   |-- ✅ source-map.js
|   |   |       |   |-- ✅ types.d-BCElaP-c.d.ts
|   |   |       |   |-- ✅ types.d.ts
|   |   |       |   \-- ✅ types.js
|   |   |       |-- ✅ diff.d.ts
|   |   |       |-- ✅ error.d.ts
|   |   |       |-- ✅ helpers.d.ts
|   |   |       |-- ✅ LICENSE
|   |   |       \-- ✅ package.json
|   |   |-- ✅ abab/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ atob.js
|   |   |   |   \-- ✅ btoa.js
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ agent-base/
|   |   |   |-- ✅ dist/
|   |   |   |   \-- ✅ src/
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ index.js.map
|   |   |   |       |-- ✅ promisify.d.ts
|   |   |   |       |-- ✅ promisify.js
|   |   |   |       \-- ✅ promisify.js.map
|   |   |   |-- ✅ src/
|   |   |   |   |-- ✅ index.ts
|   |   |   |   \-- ✅ promisify.ts
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ ansi-regex/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ ansi-styles/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ any-promise/
|   |   |   |-- ✅ register/
|   |   |   |   |-- ✅ bluebird.d.ts
|   |   |   |   |-- ✅ bluebird.js
|   |   |   |   |-- ✅ es6-promise.d.ts
|   |   |   |   |-- ✅ es6-promise.js
|   |   |   |   |-- ✅ lie.d.ts
|   |   |   |   |-- ✅ lie.js
|   |   |   |   |-- ✅ native-promise-only.d.ts
|   |   |   |   |-- ✅ native-promise-only.js
|   |   |   |   |-- ✅ pinkie.d.ts
|   |   |   |   |-- ✅ pinkie.js
|   |   |   |   |-- ✅ promise.d.ts
|   |   |   |   |-- ✅ promise.js
|   |   |   |   |-- ✅ q.d.ts
|   |   |   |   |-- ✅ q.js
|   |   |   |   |-- ✅ rsvp.d.ts
|   |   |   |   |-- ✅ rsvp.js
|   |   |   |   |-- ✅ vow.d.ts
|   |   |   |   |-- ✅ vow.js
|   |   |   |   |-- ✅ when.d.ts
|   |   |   |   \-- ✅ when.js
|   |   |   |-- ✅ .jshintrc
|   |   |   |-- ✅ .npmignore
|   |   |   |-- ✅ implementation.d.ts
|   |   |   |-- ✅ implementation.js
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ loader.js
|   |   |   |-- ✅ optional.js
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ register-shim.js
|   |   |   |-- ✅ register.d.ts
|   |   |   \-- ✅ register.js
|   |   |-- ✅ anymatch/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ arg/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ aria-query/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ etc/
|   |   |   |   |   \-- ✅ roles/
|   |   |   |   |       |-- ✅ abstract/
|   |   |   |   |       |   |-- ✅ commandRole.js
|   |   |   |   |       |   |-- ✅ compositeRole.js
|   |   |   |   |       |   |-- ✅ inputRole.js
|   |   |   |   |       |   |-- ✅ landmarkRole.js
|   |   |   |   |       |   |-- ✅ rangeRole.js
|   |   |   |   |       |   |-- ✅ roletypeRole.js
|   |   |   |   |       |   |-- ✅ sectionheadRole.js
|   |   |   |   |       |   |-- ✅ sectionRole.js
|   |   |   |   |       |   |-- ✅ selectRole.js
|   |   |   |   |       |   |-- ✅ structureRole.js
|   |   |   |   |       |   |-- ✅ widgetRole.js
|   |   |   |   |       |   \-- ✅ windowRole.js
|   |   |   |   |       |-- ✅ dpub/
|   |   |   |   |       |   |-- ✅ docAbstractRole.js
|   |   |   |   |       |   |-- ✅ docAcknowledgmentsRole.js
|   |   |   |   |       |   |-- ✅ docAfterwordRole.js
|   |   |   |   |       |   |-- ✅ docAppendixRole.js
|   |   |   |   |       |   |-- ✅ docBacklinkRole.js
|   |   |   |   |       |   |-- ✅ docBiblioentryRole.js
|   |   |   |   |       |   |-- ✅ docBibliographyRole.js
|   |   |   |   |       |   |-- ✅ docBibliorefRole.js
|   |   |   |   |       |   |-- ✅ docChapterRole.js
|   |   |   |   |       |   |-- ✅ docColophonRole.js
|   |   |   |   |       |   |-- ✅ docConclusionRole.js
|   |   |   |   |       |   |-- ✅ docCoverRole.js
|   |   |   |   |       |   |-- ✅ docCreditRole.js
|   |   |   |   |       |   |-- ✅ docCreditsRole.js
|   |   |   |   |       |   |-- ✅ docDedicationRole.js
|   |   |   |   |       |   |-- ✅ docEndnoteRole.js
|   |   |   |   |       |   |-- ✅ docEndnotesRole.js
|   |   |   |   |       |   |-- ✅ docEpigraphRole.js
|   |   |   |   |       |   |-- ✅ docEpilogueRole.js
|   |   |   |   |       |   |-- ✅ docErrataRole.js
|   |   |   |   |       |   |-- ✅ docExampleRole.js
|   |   |   |   |       |   |-- ✅ docFootnoteRole.js
|   |   |   |   |       |   |-- ✅ docForewordRole.js
|   |   |   |   |       |   |-- ✅ docGlossaryRole.js
|   |   |   |   |       |   |-- ✅ docGlossrefRole.js
|   |   |   |   |       |   |-- ✅ docIndexRole.js
|   |   |   |   |       |   |-- ✅ docIntroductionRole.js
|   |   |   |   |       |   |-- ✅ docNoterefRole.js
|   |   |   |   |       |   |-- ✅ docNoticeRole.js
|   |   |   |   |       |   |-- ✅ docPagebreakRole.js
|   |   |   |   |       |   |-- ✅ docPagefooterRole.js
|   |   |   |   |       |   |-- ✅ docPageheaderRole.js
|   |   |   |   |       |   |-- ✅ docPagelistRole.js
|   |   |   |   |       |   |-- ✅ docPartRole.js
|   |   |   |   |       |   |-- ✅ docPrefaceRole.js
|   |   |   |   |       |   |-- ✅ docPrologueRole.js
|   |   |   |   |       |   |-- ✅ docPullquoteRole.js
|   |   |   |   |       |   |-- ✅ docQnaRole.js
|   |   |   |   |       |   |-- ✅ docSubtitleRole.js
|   |   |   |   |       |   |-- ✅ docTipRole.js
|   |   |   |   |       |   \-- ✅ docTocRole.js
|   |   |   |   |       |-- ✅ graphics/
|   |   |   |   |       |   |-- ✅ graphicsDocumentRole.js
|   |   |   |   |       |   |-- ✅ graphicsObjectRole.js
|   |   |   |   |       |   \-- ✅ graphicsSymbolRole.js
|   |   |   |   |       |-- ✅ literal/
|   |   |   |   |       |   |-- ✅ alertdialogRole.js
|   |   |   |   |       |   |-- ✅ alertRole.js
|   |   |   |   |       |   |-- ✅ applicationRole.js
|   |   |   |   |       |   |-- ✅ articleRole.js
|   |   |   |   |       |   |-- ✅ bannerRole.js
|   |   |   |   |       |   |-- ✅ blockquoteRole.js
|   |   |   |   |       |   |-- ✅ buttonRole.js
|   |   |   |   |       |   |-- ✅ captionRole.js
|   |   |   |   |       |   |-- ✅ cellRole.js
|   |   |   |   |       |   |-- ✅ checkboxRole.js
|   |   |   |   |       |   |-- ✅ codeRole.js
|   |   |   |   |       |   |-- ✅ columnheaderRole.js
|   |   |   |   |       |   |-- ✅ comboboxRole.js
|   |   |   |   |       |   |-- ✅ complementaryRole.js
|   |   |   |   |       |   |-- ✅ contentinfoRole.js
|   |   |   |   |       |   |-- ✅ definitionRole.js
|   |   |   |   |       |   |-- ✅ deletionRole.js
|   |   |   |   |       |   |-- ✅ dialogRole.js
|   |   |   |   |       |   |-- ✅ directoryRole.js
|   |   |   |   |       |   |-- ✅ documentRole.js
|   |   |   |   |       |   |-- ✅ emphasisRole.js
|   |   |   |   |       |   |-- ✅ feedRole.js
|   |   |   |   |       |   |-- ✅ figureRole.js
|   |   |   |   |       |   |-- ✅ formRole.js
|   |   |   |   |       |   |-- ✅ genericRole.js
|   |   |   |   |       |   |-- ✅ graphicsDocumentRole.js
|   |   |   |   |       |   |-- ✅ graphicsObjectRole.js
|   |   |   |   |       |   |-- ✅ graphicsSymbolRole.js
|   |   |   |   |       |   |-- ✅ gridcellRole.js
|   |   |   |   |       |   |-- ✅ gridRole.js
|   |   |   |   |       |   |-- ✅ groupRole.js
|   |   |   |   |       |   |-- ✅ headingRole.js
|   |   |   |   |       |   |-- ✅ imgRole.js
|   |   |   |   |       |   |-- ✅ insertionRole.js
|   |   |   |   |       |   |-- ✅ linkRole.js
|   |   |   |   |       |   |-- ✅ listboxRole.js
|   |   |   |   |       |   |-- ✅ listitemRole.js
|   |   |   |   |       |   |-- ✅ listRole.js
|   |   |   |   |       |   |-- ✅ logRole.js
|   |   |   |   |       |   |-- ✅ mainRole.js
|   |   |   |   |       |   |-- ✅ markRole.js
|   |   |   |   |       |   |-- ✅ marqueeRole.js
|   |   |   |   |       |   |-- ✅ mathRole.js
|   |   |   |   |       |   |-- ✅ menubarRole.js
|   |   |   |   |       |   |-- ✅ menuitemcheckboxRole.js
|   |   |   |   |       |   |-- ✅ menuitemradioRole.js
|   |   |   |   |       |   |-- ✅ menuitemRole.js
|   |   |   |   |       |   |-- ✅ menuRole.js
|   |   |   |   |       |   |-- ✅ meterRole.js
|   |   |   |   |       |   |-- ✅ navigationRole.js
|   |   |   |   |       |   |-- ✅ noneRole.js
|   |   |   |   |       |   |-- ✅ noteRole.js
|   |   |   |   |       |   |-- ✅ optionRole.js
|   |   |   |   |       |   |-- ✅ paragraphRole.js
|   |   |   |   |       |   |-- ✅ presentationRole.js
|   |   |   |   |       |   |-- ✅ progressbarRole.js
|   |   |   |   |       |   |-- ✅ radiogroupRole.js
|   |   |   |   |       |   |-- ✅ radioRole.js
|   |   |   |   |       |   |-- ✅ regionRole.js
|   |   |   |   |       |   |-- ✅ rowgroupRole.js
|   |   |   |   |       |   |-- ✅ rowheaderRole.js
|   |   |   |   |       |   |-- ✅ rowRole.js
|   |   |   |   |       |   |-- ✅ scrollbarRole.js
|   |   |   |   |       |   |-- ✅ searchboxRole.js
|   |   |   |   |       |   |-- ✅ searchRole.js
|   |   |   |   |       |   |-- ✅ separatorRole.js
|   |   |   |   |       |   |-- ✅ sliderRole.js
|   |   |   |   |       |   |-- ✅ spinbuttonRole.js
|   |   |   |   |       |   |-- ✅ statusRole.js
|   |   |   |   |       |   |-- ✅ strongRole.js
|   |   |   |   |       |   |-- ✅ subscriptRole.js
|   |   |   |   |       |   |-- ✅ superscriptRole.js
|   |   |   |   |       |   |-- ✅ switchRole.js
|   |   |   |   |       |   |-- ✅ tableRole.js
|   |   |   |   |       |   |-- ✅ tablistRole.js
|   |   |   |   |       |   |-- ✅ tabpanelRole.js
|   |   |   |   |       |   |-- ✅ tabRole.js
|   |   |   |   |       |   |-- ✅ termRole.js
|   |   |   |   |       |   |-- ✅ textboxRole.js
|   |   |   |   |       |   |-- ✅ timeRole.js
|   |   |   |   |       |   |-- ✅ timerRole.js
|   |   |   |   |       |   |-- ✅ toolbarRole.js
|   |   |   |   |       |   |-- ✅ tooltipRole.js
|   |   |   |   |       |   |-- ✅ treegridRole.js
|   |   |   |   |       |   |-- ✅ treeitemRole.js
|   |   |   |   |       |   \-- ✅ treeRole.js
|   |   |   |   |       |-- ✅ ariaAbstractRoles.js
|   |   |   |   |       |-- ✅ ariaDpubRoles.js
|   |   |   |   |       |-- ✅ ariaGraphicsRoles.js
|   |   |   |   |       \-- ✅ ariaLiteralRoles.js
|   |   |   |   |-- ✅ util/
|   |   |   |   |   |-- ✅ iterationDecorator.js
|   |   |   |   |   \-- ✅ iteratorProxy.js
|   |   |   |   |-- ✅ ariaPropsMap.js
|   |   |   |   |-- ✅ domMap.js
|   |   |   |   |-- ✅ elementRoleMap.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ roleElementMap.js
|   |   |   |   \-- ✅ rolesMap.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ array-buffer-byte-length/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ assertion-error/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ asynckit/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ abort.js
|   |   |   |   |-- ✅ async.js
|   |   |   |   |-- ✅ defer.js
|   |   |   |   |-- ✅ iterate.js
|   |   |   |   |-- ✅ readable_asynckit.js
|   |   |   |   |-- ✅ readable_parallel.js
|   |   |   |   |-- ✅ readable_serial.js
|   |   |   |   |-- ✅ readable_serial_ordered.js
|   |   |   |   |-- ✅ state.js
|   |   |   |   |-- ✅ streamify.js
|   |   |   |   \-- ✅ terminator.js
|   |   |   |-- ✅ bench.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ parallel.js
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ serial.js
|   |   |   |-- ✅ serialOrdered.js
|   |   |   \-- ✅ stream.js
|   |   |-- ✅ autoprefixer/
|   |   |   |-- ✅ bin/
|   |   |   |   \-- ✅ autoprefixer
|   |   |   |-- ✅ data/
|   |   |   |   \-- ✅ prefixes.js
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ hacks/
|   |   |   |   |   |-- ✅ align-content.js
|   |   |   |   |   |-- ✅ align-items.js
|   |   |   |   |   |-- ✅ align-self.js
|   |   |   |   |   |-- ✅ animation.js
|   |   |   |   |   |-- ✅ appearance.js
|   |   |   |   |   |-- ✅ autofill.js
|   |   |   |   |   |-- ✅ backdrop-filter.js
|   |   |   |   |   |-- ✅ background-clip.js
|   |   |   |   |   |-- ✅ background-size.js
|   |   |   |   |   |-- ✅ block-logical.js
|   |   |   |   |   |-- ✅ border-image.js
|   |   |   |   |   |-- ✅ border-radius.js
|   |   |   |   |   |-- ✅ break-props.js
|   |   |   |   |   |-- ✅ cross-fade.js
|   |   |   |   |   |-- ✅ display-flex.js
|   |   |   |   |   |-- ✅ display-grid.js
|   |   |   |   |   |-- ✅ file-selector-button.js
|   |   |   |   |   |-- ✅ filter-value.js
|   |   |   |   |   |-- ✅ filter.js
|   |   |   |   |   |-- ✅ flex-basis.js
|   |   |   |   |   |-- ✅ flex-direction.js
|   |   |   |   |   |-- ✅ flex-flow.js
|   |   |   |   |   |-- ✅ flex-grow.js
|   |   |   |   |   |-- ✅ flex-shrink.js
|   |   |   |   |   |-- ✅ flex-spec.js
|   |   |   |   |   |-- ✅ flex-wrap.js
|   |   |   |   |   |-- ✅ flex.js
|   |   |   |   |   |-- ✅ fullscreen.js
|   |   |   |   |   |-- ✅ gradient.js
|   |   |   |   |   |-- ✅ grid-area.js
|   |   |   |   |   |-- ✅ grid-column-align.js
|   |   |   |   |   |-- ✅ grid-end.js
|   |   |   |   |   |-- ✅ grid-row-align.js
|   |   |   |   |   |-- ✅ grid-row-column.js
|   |   |   |   |   |-- ✅ grid-rows-columns.js
|   |   |   |   |   |-- ✅ grid-start.js
|   |   |   |   |   |-- ✅ grid-template-areas.js
|   |   |   |   |   |-- ✅ grid-template.js
|   |   |   |   |   |-- ✅ grid-utils.js
|   |   |   |   |   |-- ✅ image-rendering.js
|   |   |   |   |   |-- ✅ image-set.js
|   |   |   |   |   |-- ✅ inline-logical.js
|   |   |   |   |   |-- ✅ intrinsic.js
|   |   |   |   |   |-- ✅ justify-content.js
|   |   |   |   |   |-- ✅ mask-border.js
|   |   |   |   |   |-- ✅ mask-composite.js
|   |   |   |   |   |-- ✅ order.js
|   |   |   |   |   |-- ✅ overscroll-behavior.js
|   |   |   |   |   |-- ✅ pixelated.js
|   |   |   |   |   |-- ✅ place-self.js
|   |   |   |   |   |-- ✅ placeholder-shown.js
|   |   |   |   |   |-- ✅ placeholder.js
|   |   |   |   |   |-- ✅ print-color-adjust.js
|   |   |   |   |   |-- ✅ text-decoration-skip-ink.js
|   |   |   |   |   |-- ✅ text-decoration.js
|   |   |   |   |   |-- ✅ text-emphasis-position.js
|   |   |   |   |   |-- ✅ transform-decl.js
|   |   |   |   |   |-- ✅ user-select.js
|   |   |   |   |   \-- ✅ writing-mode.js
|   |   |   |   |-- ✅ at-rule.js
|   |   |   |   |-- ✅ autoprefixer.d.ts
|   |   |   |   |-- ✅ autoprefixer.js
|   |   |   |   |-- ✅ brackets.js
|   |   |   |   |-- ✅ browsers.js
|   |   |   |   |-- ✅ declaration.js
|   |   |   |   |-- ✅ info.js
|   |   |   |   |-- ✅ old-selector.js
|   |   |   |   |-- ✅ old-value.js
|   |   |   |   |-- ✅ prefixer.js
|   |   |   |   |-- ✅ prefixes.js
|   |   |   |   |-- ✅ processor.js
|   |   |   |   |-- ✅ resolution.js
|   |   |   |   |-- ✅ selector.js
|   |   |   |   |-- ✅ supports.js
|   |   |   |   |-- ✅ transition.js
|   |   |   |   |-- ✅ utils.js
|   |   |   |   |-- ✅ value.js
|   |   |   |   \-- ✅ vendor.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ available-typed-arrays/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ axios/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ browser/
|   |   |   |   |   |-- ✅ axios.cjs
|   |   |   |   |   \-- ✅ axios.cjs.map
|   |   |   |   |-- ✅ esm/
|   |   |   |   |   |-- ✅ axios.js
|   |   |   |   |   |-- ✅ axios.js.map
|   |   |   |   |   |-- ✅ axios.min.js
|   |   |   |   |   \-- ✅ axios.min.js.map
|   |   |   |   |-- ✅ node/
|   |   |   |   |   |-- ✅ axios.cjs
|   |   |   |   |   \-- ✅ axios.cjs.map
|   |   |   |   |-- ✅ axios.js
|   |   |   |   |-- ✅ axios.js.map
|   |   |   |   |-- ✅ axios.min.js
|   |   |   |   \-- ✅ axios.min.js.map
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ adapters/
|   |   |   |   |   |-- ✅ adapters.js
|   |   |   |   |   |-- ✅ fetch.js
|   |   |   |   |   |-- ✅ http.js
|   |   |   |   |   |-- ✅ README.md
|   |   |   |   |   \-- ✅ xhr.js
|   |   |   |   |-- ✅ cancel/
|   |   |   |   |   |-- ✅ CanceledError.js
|   |   |   |   |   |-- ✅ CancelToken.js
|   |   |   |   |   \-- ✅ isCancel.js
|   |   |   |   |-- ✅ core/
|   |   |   |   |   |-- ✅ Axios.js
|   |   |   |   |   |-- ✅ AxiosError.js
|   |   |   |   |   |-- ✅ AxiosHeaders.js
|   |   |   |   |   |-- ✅ buildFullPath.js
|   |   |   |   |   |-- ✅ dispatchRequest.js
|   |   |   |   |   |-- ✅ InterceptorManager.js
|   |   |   |   |   |-- ✅ mergeConfig.js
|   |   |   |   |   |-- ✅ README.md
|   |   |   |   |   |-- ✅ settle.js
|   |   |   |   |   \-- ✅ transformData.js
|   |   |   |   |-- ✅ defaults/
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   \-- ✅ transitional.js
|   |   |   |   |-- ✅ env/
|   |   |   |   |   |-- ✅ classes/
|   |   |   |   |   |   \-- ✅ FormData.js
|   |   |   |   |   |-- ✅ data.js
|   |   |   |   |   \-- ✅ README.md
|   |   |   |   |-- ✅ helpers/
|   |   |   |   |   |-- ✅ AxiosTransformStream.js
|   |   |   |   |   |-- ✅ AxiosURLSearchParams.js
|   |   |   |   |   |-- ✅ bind.js
|   |   |   |   |   |-- ✅ buildURL.js
|   |   |   |   |   |-- ✅ callbackify.js
|   |   |   |   |   |-- ✅ combineURLs.js
|   |   |   |   |   |-- ✅ composeSignals.js
|   |   |   |   |   |-- ✅ cookies.js
|   |   |   |   |   |-- ✅ deprecatedMethod.js
|   |   |   |   |   |-- ✅ estimateDataURLDecodedBytes.js
|   |   |   |   |   |-- ✅ formDataToJSON.js
|   |   |   |   |   |-- ✅ formDataToStream.js
|   |   |   |   |   |-- ✅ fromDataURI.js
|   |   |   |   |   |-- ✅ HttpStatusCode.js
|   |   |   |   |   |-- ✅ isAbsoluteURL.js
|   |   |   |   |   |-- ✅ isAxiosError.js
|   |   |   |   |   |-- ✅ isURLSameOrigin.js
|   |   |   |   |   |-- ✅ null.js
|   |   |   |   |   |-- ✅ parseHeaders.js
|   |   |   |   |   |-- ✅ parseProtocol.js
|   |   |   |   |   |-- ✅ progressEventReducer.js
|   |   |   |   |   |-- ✅ readBlob.js
|   |   |   |   |   |-- ✅ README.md
|   |   |   |   |   |-- ✅ resolveConfig.js
|   |   |   |   |   |-- ✅ speedometer.js
|   |   |   |   |   |-- ✅ spread.js
|   |   |   |   |   |-- ✅ throttle.js
|   |   |   |   |   |-- ✅ toFormData.js
|   |   |   |   |   |-- ✅ toURLEncodedForm.js
|   |   |   |   |   |-- ✅ trackStream.js
|   |   |   |   |   |-- ✅ validator.js
|   |   |   |   |   \-- ✅ ZlibHeaderTransformStream.js
|   |   |   |   |-- ✅ platform/
|   |   |   |   |   |-- ✅ browser/
|   |   |   |   |   |   |-- ✅ classes/
|   |   |   |   |   |   |   |-- ✅ Blob.js
|   |   |   |   |   |   |   |-- ✅ FormData.js
|   |   |   |   |   |   |   \-- ✅ URLSearchParams.js
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   |-- ✅ common/
|   |   |   |   |   |   \-- ✅ utils.js
|   |   |   |   |   |-- ✅ node/
|   |   |   |   |   |   |-- ✅ classes/
|   |   |   |   |   |   |   |-- ✅ FormData.js
|   |   |   |   |   |   |   \-- ✅ URLSearchParams.js
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ axios.js
|   |   |   |   \-- ✅ utils.js
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.cts
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ MIGRATION_GUIDE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ balanced-match/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ baseline-browser-mapping/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ cli.js
|   |   |   |   |-- ✅ index.cjs
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ binary-extensions/
|   |   |   |-- ✅ binary-extensions.json
|   |   |   |-- ✅ binary-extensions.json.d.ts
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ brace-expansion/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ braces/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ compile.js
|   |   |   |   |-- ✅ constants.js
|   |   |   |   |-- ✅ expand.js
|   |   |   |   |-- ✅ parse.js
|   |   |   |   |-- ✅ stringify.js
|   |   |   |   \-- ✅ utils.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ browserslist/
|   |   |   |-- ✅ browser.js
|   |   |   |-- ✅ cli.js
|   |   |   |-- ✅ error.d.ts
|   |   |   |-- ✅ error.js
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ node.js
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ parse.js
|   |   |   \-- ✅ README.md
|   |   |-- ✅ cac/
|   |   |   |-- ✅ deno/
|   |   |   |   |-- ✅ CAC.ts
|   |   |   |   |-- ✅ Command.ts
|   |   |   |   |-- ✅ deno.ts
|   |   |   |   |-- ✅ index.ts
|   |   |   |   |-- ✅ Option.ts
|   |   |   |   \-- ✅ utils.ts
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ index.mjs
|   |   |   |-- ✅ index-compat.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ mod.js
|   |   |   |-- ✅ mod.ts
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ call-bind/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ callBound.js
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintignore
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ callBound.js
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ call-bind-apply-helpers/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ actualApply.d.ts
|   |   |   |-- ✅ actualApply.js
|   |   |   |-- ✅ applyBind.d.ts
|   |   |   |-- ✅ applyBind.js
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ functionApply.d.ts
|   |   |   |-- ✅ functionApply.js
|   |   |   |-- ✅ functionCall.d.ts
|   |   |   |-- ✅ functionCall.js
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ reflectApply.d.ts
|   |   |   |-- ✅ reflectApply.js
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ call-bound/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ camelcase-css/
|   |   |   |-- ✅ index-es5.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ caniuse-lite/
|   |   |   |-- ✅ data/
|   |   |   |   |-- ✅ features/
|   |   |   |   |   |-- ✅ aac.js
|   |   |   |   |   |-- ✅ abortcontroller.js
|   |   |   |   |   |-- ✅ ac3-ec3.js
|   |   |   |   |   |-- ✅ accelerometer.js
|   |   |   |   |   |-- ✅ addeventlistener.js
|   |   |   |   |   |-- ✅ alternate-stylesheet.js
|   |   |   |   |   |-- ✅ ambient-light.js
|   |   |   |   |   |-- ✅ apng.js
|   |   |   |   |   |-- ✅ array-find-index.js
|   |   |   |   |   |-- ✅ array-find.js
|   |   |   |   |   |-- ✅ array-flat.js
|   |   |   |   |   |-- ✅ array-includes.js
|   |   |   |   |   |-- ✅ arrow-functions.js
|   |   |   |   |   |-- ✅ asmjs.js
|   |   |   |   |   |-- ✅ async-clipboard.js
|   |   |   |   |   |-- ✅ async-functions.js
|   |   |   |   |   |-- ✅ atob-btoa.js
|   |   |   |   |   |-- ✅ audio-api.js
|   |   |   |   |   |-- ✅ audio.js
|   |   |   |   |   |-- ✅ audiotracks.js
|   |   |   |   |   |-- ✅ autofocus.js
|   |   |   |   |   |-- ✅ auxclick.js
|   |   |   |   |   |-- ✅ av1.js
|   |   |   |   |   |-- ✅ avif.js
|   |   |   |   |   |-- ✅ background-attachment.js
|   |   |   |   |   |-- ✅ background-clip-text.js
|   |   |   |   |   |-- ✅ background-img-opts.js
|   |   |   |   |   |-- ✅ background-position-x-y.js
|   |   |   |   |   |-- ✅ background-repeat-round-space.js
|   |   |   |   |   |-- ✅ background-sync.js
|   |   |   |   |   |-- ✅ battery-status.js
|   |   |   |   |   |-- ✅ beacon.js
|   |   |   |   |   |-- ✅ beforeafterprint.js
|   |   |   |   |   |-- ✅ bigint.js
|   |   |   |   |   |-- ✅ blobbuilder.js
|   |   |   |   |   |-- ✅ bloburls.js
|   |   |   |   |   |-- ✅ border-image.js
|   |   |   |   |   |-- ✅ border-radius.js
|   |   |   |   |   |-- ✅ broadcastchannel.js
|   |   |   |   |   |-- ✅ brotli.js
|   |   |   |   |   |-- ✅ calc.js
|   |   |   |   |   |-- ✅ canvas-blending.js
|   |   |   |   |   |-- ✅ canvas-text.js
|   |   |   |   |   |-- ✅ canvas.js
|   |   |   |   |   |-- ✅ ch-unit.js
|   |   |   |   |   |-- ✅ chacha20-poly1305.js
|   |   |   |   |   |-- ✅ channel-messaging.js
|   |   |   |   |   |-- ✅ childnode-remove.js
|   |   |   |   |   |-- ✅ classlist.js
|   |   |   |   |   |-- ✅ client-hints-dpr-width-viewport.js
|   |   |   |   |   |-- ✅ clipboard.js
|   |   |   |   |   |-- ✅ colr-v1.js
|   |   |   |   |   |-- ✅ colr.js
|   |   |   |   |   |-- ✅ comparedocumentposition.js
|   |   |   |   |   |-- ✅ console-basic.js
|   |   |   |   |   |-- ✅ console-time.js
|   |   |   |   |   |-- ✅ const.js
|   |   |   |   |   |-- ✅ constraint-validation.js
|   |   |   |   |   |-- ✅ contenteditable.js
|   |   |   |   |   |-- ✅ contentsecuritypolicy.js
|   |   |   |   |   |-- ✅ contentsecuritypolicy2.js
|   |   |   |   |   |-- ✅ cookie-store-api.js
|   |   |   |   |   |-- ✅ cors.js
|   |   |   |   |   |-- ✅ createimagebitmap.js
|   |   |   |   |   |-- ✅ credential-management.js
|   |   |   |   |   |-- ✅ cross-document-view-transitions.js
|   |   |   |   |   |-- ✅ cryptography.js
|   |   |   |   |   |-- ✅ css-all.js
|   |   |   |   |   |-- ✅ css-anchor-positioning.js
|   |   |   |   |   |-- ✅ css-animation.js
|   |   |   |   |   |-- ✅ css-any-link.js
|   |   |   |   |   |-- ✅ css-appearance.js
|   |   |   |   |   |-- ✅ css-at-counter-style.js
|   |   |   |   |   |-- ✅ css-autofill.js
|   |   |   |   |   |-- ✅ css-backdrop-filter.js
|   |   |   |   |   |-- ✅ css-background-offsets.js
|   |   |   |   |   |-- ✅ css-backgroundblendmode.js
|   |   |   |   |   |-- ✅ css-boxdecorationbreak.js
|   |   |   |   |   |-- ✅ css-boxshadow.js
|   |   |   |   |   |-- ✅ css-canvas.js
|   |   |   |   |   |-- ✅ css-caret-color.js
|   |   |   |   |   |-- ✅ css-cascade-layers.js
|   |   |   |   |   |-- ✅ css-cascade-scope.js
|   |   |   |   |   |-- ✅ css-case-insensitive.js
|   |   |   |   |   |-- ✅ css-clip-path.js
|   |   |   |   |   |-- ✅ css-color-adjust.js
|   |   |   |   |   |-- ✅ css-color-function.js
|   |   |   |   |   |-- ✅ css-conic-gradients.js
|   |   |   |   |   |-- ✅ css-container-queries-style.js
|   |   |   |   |   |-- ✅ css-container-queries.js
|   |   |   |   |   |-- ✅ css-container-query-units.js
|   |   |   |   |   |-- ✅ css-containment.js
|   |   |   |   |   |-- ✅ css-content-visibility.js
|   |   |   |   |   |-- ✅ css-counters.js
|   |   |   |   |   |-- ✅ css-crisp-edges.js
|   |   |   |   |   |-- ✅ css-cross-fade.js
|   |   |   |   |   |-- ✅ css-default-pseudo.js
|   |   |   |   |   |-- ✅ css-descendant-gtgt.js
|   |   |   |   |   |-- ✅ css-deviceadaptation.js
|   |   |   |   |   |-- ✅ css-dir-pseudo.js
|   |   |   |   |   |-- ✅ css-display-contents.js
|   |   |   |   |   |-- ✅ css-element-function.js
|   |   |   |   |   |-- ✅ css-env-function.js
|   |   |   |   |   |-- ✅ css-exclusions.js
|   |   |   |   |   |-- ✅ css-featurequeries.js
|   |   |   |   |   |-- ✅ css-file-selector-button.js
|   |   |   |   |   |-- ✅ css-filter-function.js
|   |   |   |   |   |-- ✅ css-filters.js
|   |   |   |   |   |-- ✅ css-first-letter.js
|   |   |   |   |   |-- ✅ css-first-line.js
|   |   |   |   |   |-- ✅ css-fixed.js
|   |   |   |   |   |-- ✅ css-focus-visible.js
|   |   |   |   |   |-- ✅ css-focus-within.js
|   |   |   |   |   |-- ✅ css-font-palette.js
|   |   |   |   |   |-- ✅ css-font-rendering-controls.js
|   |   |   |   |   |-- ✅ css-font-stretch.js
|   |   |   |   |   |-- ✅ css-gencontent.js
|   |   |   |   |   |-- ✅ css-gradients.js
|   |   |   |   |   |-- ✅ css-grid-animation.js
|   |   |   |   |   |-- ✅ css-grid.js
|   |   |   |   |   |-- ✅ css-hanging-punctuation.js
|   |   |   |   |   |-- ✅ css-has.js
|   |   |   |   |   |-- ✅ css-hyphens.js
|   |   |   |   |   |-- ✅ css-if.js
|   |   |   |   |   |-- ✅ css-image-orientation.js
|   |   |   |   |   |-- ✅ css-image-set.js
|   |   |   |   |   |-- ✅ css-in-out-of-range.js
|   |   |   |   |   |-- ✅ css-indeterminate-pseudo.js
|   |   |   |   |   |-- ✅ css-initial-letter.js
|   |   |   |   |   |-- ✅ css-initial-value.js
|   |   |   |   |   |-- ✅ css-lch-lab.js
|   |   |   |   |   |-- ✅ css-letter-spacing.js
|   |   |   |   |   |-- ✅ css-line-clamp.js
|   |   |   |   |   |-- ✅ css-logical-props.js
|   |   |   |   |   |-- ✅ css-marker-pseudo.js
|   |   |   |   |   |-- ✅ css-masks.js
|   |   |   |   |   |-- ✅ css-matches-pseudo.js
|   |   |   |   |   |-- ✅ css-math-functions.js
|   |   |   |   |   |-- ✅ css-media-interaction.js
|   |   |   |   |   |-- ✅ css-media-range-syntax.js
|   |   |   |   |   |-- ✅ css-media-resolution.js
|   |   |   |   |   |-- ✅ css-media-scripting.js
|   |   |   |   |   |-- ✅ css-mediaqueries.js
|   |   |   |   |   |-- ✅ css-mixblendmode.js
|   |   |   |   |   |-- ✅ css-module-scripts.js
|   |   |   |   |   |-- ✅ css-motion-paths.js
|   |   |   |   |   |-- ✅ css-namespaces.js
|   |   |   |   |   |-- ✅ css-nesting.js
|   |   |   |   |   |-- ✅ css-not-sel-list.js
|   |   |   |   |   |-- ✅ css-nth-child-of.js
|   |   |   |   |   |-- ✅ css-opacity.js
|   |   |   |   |   |-- ✅ css-optional-pseudo.js
|   |   |   |   |   |-- ✅ css-overflow-anchor.js
|   |   |   |   |   |-- ✅ css-overflow-overlay.js
|   |   |   |   |   |-- ✅ css-overflow.js
|   |   |   |   |   |-- ✅ css-overscroll-behavior.js
|   |   |   |   |   |-- ✅ css-page-break.js
|   |   |   |   |   |-- ✅ css-paged-media.js
|   |   |   |   |   |-- ✅ css-paint-api.js
|   |   |   |   |   |-- ✅ css-placeholder-shown.js
|   |   |   |   |   |-- ✅ css-placeholder.js
|   |   |   |   |   |-- ✅ css-print-color-adjust.js
|   |   |   |   |   |-- ✅ css-read-only-write.js
|   |   |   |   |   |-- ✅ css-rebeccapurple.js
|   |   |   |   |   |-- ✅ css-reflections.js
|   |   |   |   |   |-- ✅ css-regions.js
|   |   |   |   |   |-- ✅ css-relative-colors.js
|   |   |   |   |   |-- ✅ css-repeating-gradients.js
|   |   |   |   |   |-- ✅ css-resize.js
|   |   |   |   |   |-- ✅ css-revert-value.js
|   |   |   |   |   |-- ✅ css-rrggbbaa.js
|   |   |   |   |   |-- ✅ css-scroll-behavior.js
|   |   |   |   |   |-- ✅ css-scrollbar.js
|   |   |   |   |   |-- ✅ css-sel2.js
|   |   |   |   |   |-- ✅ css-sel3.js
|   |   |   |   |   |-- ✅ css-selection.js
|   |   |   |   |   |-- ✅ css-shapes.js
|   |   |   |   |   |-- ✅ css-snappoints.js
|   |   |   |   |   |-- ✅ css-sticky.js
|   |   |   |   |   |-- ✅ css-subgrid.js
|   |   |   |   |   |-- ✅ css-supports-api.js
|   |   |   |   |   |-- ✅ css-table.js
|   |   |   |   |   |-- ✅ css-text-align-last.js
|   |   |   |   |   |-- ✅ css-text-box-trim.js
|   |   |   |   |   |-- ✅ css-text-indent.js
|   |   |   |   |   |-- ✅ css-text-justify.js
|   |   |   |   |   |-- ✅ css-text-orientation.js
|   |   |   |   |   |-- ✅ css-text-spacing.js
|   |   |   |   |   |-- ✅ css-text-wrap-balance.js
|   |   |   |   |   |-- ✅ css-textshadow.js
|   |   |   |   |   |-- ✅ css-touch-action.js
|   |   |   |   |   |-- ✅ css-transitions.js
|   |   |   |   |   |-- ✅ css-unicode-bidi.js
|   |   |   |   |   |-- ✅ css-unset-value.js
|   |   |   |   |   |-- ✅ css-variables.js
|   |   |   |   |   |-- ✅ css-when-else.js
|   |   |   |   |   |-- ✅ css-widows-orphans.js
|   |   |   |   |   |-- ✅ css-width-stretch.js
|   |   |   |   |   |-- ✅ css-writing-mode.js
|   |   |   |   |   |-- ✅ css-zoom.js
|   |   |   |   |   |-- ✅ css3-attr.js
|   |   |   |   |   |-- ✅ css3-boxsizing.js
|   |   |   |   |   |-- ✅ css3-colors.js
|   |   |   |   |   |-- ✅ css3-cursors-grab.js
|   |   |   |   |   |-- ✅ css3-cursors-newer.js
|   |   |   |   |   |-- ✅ css3-cursors.js
|   |   |   |   |   |-- ✅ css3-tabsize.js
|   |   |   |   |   |-- ✅ currentcolor.js
|   |   |   |   |   |-- ✅ custom-elements.js
|   |   |   |   |   |-- ✅ custom-elementsv1.js
|   |   |   |   |   |-- ✅ customevent.js
|   |   |   |   |   |-- ✅ datalist.js
|   |   |   |   |   |-- ✅ dataset.js
|   |   |   |   |   |-- ✅ datauri.js
|   |   |   |   |   |-- ✅ date-tolocaledatestring.js
|   |   |   |   |   |-- ✅ declarative-shadow-dom.js
|   |   |   |   |   |-- ✅ decorators.js
|   |   |   |   |   |-- ✅ details.js
|   |   |   |   |   |-- ✅ deviceorientation.js
|   |   |   |   |   |-- ✅ devicepixelratio.js
|   |   |   |   |   |-- ✅ dialog.js
|   |   |   |   |   |-- ✅ dispatchevent.js
|   |   |   |   |   |-- ✅ dnssec.js
|   |   |   |   |   |-- ✅ do-not-track.js
|   |   |   |   |   |-- ✅ document-currentscript.js
|   |   |   |   |   |-- ✅ document-evaluate-xpath.js
|   |   |   |   |   |-- ✅ document-execcommand.js
|   |   |   |   |   |-- ✅ document-policy.js
|   |   |   |   |   |-- ✅ document-scrollingelement.js
|   |   |   |   |   |-- ✅ documenthead.js
|   |   |   |   |   |-- ✅ dom-manip-convenience.js
|   |   |   |   |   |-- ✅ dom-range.js
|   |   |   |   |   |-- ✅ domcontentloaded.js
|   |   |   |   |   |-- ✅ dommatrix.js
|   |   |   |   |   |-- ✅ download.js
|   |   |   |   |   |-- ✅ dragndrop.js
|   |   |   |   |   |-- ✅ element-closest.js
|   |   |   |   |   |-- ✅ element-from-point.js
|   |   |   |   |   |-- ✅ element-scroll-methods.js
|   |   |   |   |   |-- ✅ eme.js
|   |   |   |   |   |-- ✅ eot.js
|   |   |   |   |   |-- ✅ es5.js
|   |   |   |   |   |-- ✅ es6-class.js
|   |   |   |   |   |-- ✅ es6-generators.js
|   |   |   |   |   |-- ✅ es6-module-dynamic-import.js
|   |   |   |   |   |-- ✅ es6-module.js
|   |   |   |   |   |-- ✅ es6-number.js
|   |   |   |   |   |-- ✅ es6-string-includes.js
|   |   |   |   |   |-- ✅ es6.js
|   |   |   |   |   |-- ✅ eventsource.js
|   |   |   |   |   |-- ✅ extended-system-fonts.js
|   |   |   |   |   |-- ✅ feature-policy.js
|   |   |   |   |   |-- ✅ fetch.js
|   |   |   |   |   |-- ✅ fieldset-disabled.js
|   |   |   |   |   |-- ✅ fileapi.js
|   |   |   |   |   |-- ✅ filereader.js
|   |   |   |   |   |-- ✅ filereadersync.js
|   |   |   |   |   |-- ✅ filesystem.js
|   |   |   |   |   |-- ✅ flac.js
|   |   |   |   |   |-- ✅ flexbox-gap.js
|   |   |   |   |   |-- ✅ flexbox.js
|   |   |   |   |   |-- ✅ flow-root.js
|   |   |   |   |   |-- ✅ focusin-focusout-events.js
|   |   |   |   |   |-- ✅ font-family-system-ui.js
|   |   |   |   |   |-- ✅ font-feature.js
|   |   |   |   |   |-- ✅ font-kerning.js
|   |   |   |   |   |-- ✅ font-loading.js
|   |   |   |   |   |-- ✅ font-size-adjust.js
|   |   |   |   |   |-- ✅ font-smooth.js
|   |   |   |   |   |-- ✅ font-unicode-range.js
|   |   |   |   |   |-- ✅ font-variant-alternates.js
|   |   |   |   |   |-- ✅ font-variant-numeric.js
|   |   |   |   |   |-- ✅ fontface.js
|   |   |   |   |   |-- ✅ form-attribute.js
|   |   |   |   |   |-- ✅ form-submit-attributes.js
|   |   |   |   |   |-- ✅ form-validation.js
|   |   |   |   |   |-- ✅ forms.js
|   |   |   |   |   |-- ✅ fullscreen.js
|   |   |   |   |   |-- ✅ gamepad.js
|   |   |   |   |   |-- ✅ geolocation.js
|   |   |   |   |   |-- ✅ getboundingclientrect.js
|   |   |   |   |   |-- ✅ getcomputedstyle.js
|   |   |   |   |   |-- ✅ getelementsbyclassname.js
|   |   |   |   |   |-- ✅ getrandomvalues.js
|   |   |   |   |   |-- ✅ gyroscope.js
|   |   |   |   |   |-- ✅ hardwareconcurrency.js
|   |   |   |   |   |-- ✅ hashchange.js
|   |   |   |   |   |-- ✅ heif.js
|   |   |   |   |   |-- ✅ hevc.js
|   |   |   |   |   |-- ✅ hidden.js
|   |   |   |   |   |-- ✅ high-resolution-time.js
|   |   |   |   |   |-- ✅ history.js
|   |   |   |   |   |-- ✅ html-media-capture.js
|   |   |   |   |   |-- ✅ html5semantic.js
|   |   |   |   |   |-- ✅ http-live-streaming.js
|   |   |   |   |   |-- ✅ http2.js
|   |   |   |   |   |-- ✅ http3.js
|   |   |   |   |   |-- ✅ iframe-sandbox.js
|   |   |   |   |   |-- ✅ iframe-seamless.js
|   |   |   |   |   |-- ✅ iframe-srcdoc.js
|   |   |   |   |   |-- ✅ imagecapture.js
|   |   |   |   |   |-- ✅ ime.js
|   |   |   |   |   |-- ✅ img-naturalwidth-naturalheight.js
|   |   |   |   |   |-- ✅ import-maps.js
|   |   |   |   |   |-- ✅ imports.js
|   |   |   |   |   |-- ✅ indeterminate-checkbox.js
|   |   |   |   |   |-- ✅ indexeddb.js
|   |   |   |   |   |-- ✅ indexeddb2.js
|   |   |   |   |   |-- ✅ inline-block.js
|   |   |   |   |   |-- ✅ innertext.js
|   |   |   |   |   |-- ✅ input-autocomplete-onoff.js
|   |   |   |   |   |-- ✅ input-color.js
|   |   |   |   |   |-- ✅ input-datetime.js
|   |   |   |   |   |-- ✅ input-email-tel-url.js
|   |   |   |   |   |-- ✅ input-event.js
|   |   |   |   |   |-- ✅ input-file-accept.js
|   |   |   |   |   |-- ✅ input-file-directory.js
|   |   |   |   |   |-- ✅ input-file-multiple.js
|   |   |   |   |   |-- ✅ input-inputmode.js
|   |   |   |   |   |-- ✅ input-minlength.js
|   |   |   |   |   |-- ✅ input-number.js
|   |   |   |   |   |-- ✅ input-pattern.js
|   |   |   |   |   |-- ✅ input-placeholder.js
|   |   |   |   |   |-- ✅ input-range.js
|   |   |   |   |   |-- ✅ input-search.js
|   |   |   |   |   |-- ✅ input-selection.js
|   |   |   |   |   |-- ✅ insert-adjacent.js
|   |   |   |   |   |-- ✅ insertadjacenthtml.js
|   |   |   |   |   |-- ✅ internationalization.js
|   |   |   |   |   |-- ✅ intersectionobserver-v2.js
|   |   |   |   |   |-- ✅ intersectionobserver.js
|   |   |   |   |   |-- ✅ intl-pluralrules.js
|   |   |   |   |   |-- ✅ intrinsic-width.js
|   |   |   |   |   |-- ✅ jpeg2000.js
|   |   |   |   |   |-- ✅ jpegxl.js
|   |   |   |   |   |-- ✅ jpegxr.js
|   |   |   |   |   |-- ✅ js-regexp-lookbehind.js
|   |   |   |   |   |-- ✅ json.js
|   |   |   |   |   |-- ✅ justify-content-space-evenly.js
|   |   |   |   |   |-- ✅ kerning-pairs-ligatures.js
|   |   |   |   |   |-- ✅ keyboardevent-charcode.js
|   |   |   |   |   |-- ✅ keyboardevent-code.js
|   |   |   |   |   |-- ✅ keyboardevent-getmodifierstate.js
|   |   |   |   |   |-- ✅ keyboardevent-key.js
|   |   |   |   |   |-- ✅ keyboardevent-location.js
|   |   |   |   |   |-- ✅ keyboardevent-which.js
|   |   |   |   |   |-- ✅ lazyload.js
|   |   |   |   |   |-- ✅ let.js
|   |   |   |   |   |-- ✅ link-icon-png.js
|   |   |   |   |   |-- ✅ link-icon-svg.js
|   |   |   |   |   |-- ✅ link-rel-dns-prefetch.js
|   |   |   |   |   |-- ✅ link-rel-modulepreload.js
|   |   |   |   |   |-- ✅ link-rel-preconnect.js
|   |   |   |   |   |-- ✅ link-rel-prefetch.js
|   |   |   |   |   |-- ✅ link-rel-preload.js
|   |   |   |   |   |-- ✅ link-rel-prerender.js
|   |   |   |   |   |-- ✅ loading-lazy-attr.js
|   |   |   |   |   |-- ✅ localecompare.js
|   |   |   |   |   |-- ✅ magnetometer.js
|   |   |   |   |   |-- ✅ matchesselector.js
|   |   |   |   |   |-- ✅ matchmedia.js
|   |   |   |   |   |-- ✅ mathml.js
|   |   |   |   |   |-- ✅ maxlength.js
|   |   |   |   |   |-- ✅ mdn-css-backdrop-pseudo-element.js
|   |   |   |   |   |-- ✅ mdn-css-unicode-bidi-isolate-override.js
|   |   |   |   |   |-- ✅ mdn-css-unicode-bidi-isolate.js
|   |   |   |   |   |-- ✅ mdn-css-unicode-bidi-plaintext.js
|   |   |   |   |   |-- ✅ mdn-text-decoration-color.js
|   |   |   |   |   |-- ✅ mdn-text-decoration-line.js
|   |   |   |   |   |-- ✅ mdn-text-decoration-shorthand.js
|   |   |   |   |   |-- ✅ mdn-text-decoration-style.js
|   |   |   |   |   |-- ✅ media-fragments.js
|   |   |   |   |   |-- ✅ mediacapture-fromelement.js
|   |   |   |   |   |-- ✅ mediarecorder.js
|   |   |   |   |   |-- ✅ mediasource.js
|   |   |   |   |   |-- ✅ menu.js
|   |   |   |   |   |-- ✅ meta-theme-color.js
|   |   |   |   |   |-- ✅ meter.js
|   |   |   |   |   |-- ✅ midi.js
|   |   |   |   |   |-- ✅ minmaxwh.js
|   |   |   |   |   |-- ✅ mp3.js
|   |   |   |   |   |-- ✅ mpeg-dash.js
|   |   |   |   |   |-- ✅ mpeg4.js
|   |   |   |   |   |-- ✅ multibackgrounds.js
|   |   |   |   |   |-- ✅ multicolumn.js
|   |   |   |   |   |-- ✅ mutation-events.js
|   |   |   |   |   |-- ✅ mutationobserver.js
|   |   |   |   |   |-- ✅ namevalue-storage.js
|   |   |   |   |   |-- ✅ native-filesystem-api.js
|   |   |   |   |   |-- ✅ nav-timing.js
|   |   |   |   |   |-- ✅ netinfo.js
|   |   |   |   |   |-- ✅ notifications.js
|   |   |   |   |   |-- ✅ object-entries.js
|   |   |   |   |   |-- ✅ object-fit.js
|   |   |   |   |   |-- ✅ object-observe.js
|   |   |   |   |   |-- ✅ object-values.js
|   |   |   |   |   |-- ✅ objectrtc.js
|   |   |   |   |   |-- ✅ offline-apps.js
|   |   |   |   |   |-- ✅ offscreencanvas.js
|   |   |   |   |   |-- ✅ ogg-vorbis.js
|   |   |   |   |   |-- ✅ ogv.js
|   |   |   |   |   |-- ✅ ol-reversed.js
|   |   |   |   |   |-- ✅ once-event-listener.js
|   |   |   |   |   |-- ✅ online-status.js
|   |   |   |   |   |-- ✅ opus.js
|   |   |   |   |   |-- ✅ orientation-sensor.js
|   |   |   |   |   |-- ✅ outline.js
|   |   |   |   |   |-- ✅ pad-start-end.js
|   |   |   |   |   |-- ✅ page-transition-events.js
|   |   |   |   |   |-- ✅ pagevisibility.js
|   |   |   |   |   |-- ✅ passive-event-listener.js
|   |   |   |   |   |-- ✅ passkeys.js
|   |   |   |   |   |-- ✅ passwordrules.js
|   |   |   |   |   |-- ✅ path2d.js
|   |   |   |   |   |-- ✅ payment-request.js
|   |   |   |   |   |-- ✅ pdf-viewer.js
|   |   |   |   |   |-- ✅ permissions-api.js
|   |   |   |   |   |-- ✅ permissions-policy.js
|   |   |   |   |   |-- ✅ picture-in-picture.js
|   |   |   |   |   |-- ✅ picture.js
|   |   |   |   |   |-- ✅ ping.js
|   |   |   |   |   |-- ✅ png-alpha.js
|   |   |   |   |   |-- ✅ pointer-events.js
|   |   |   |   |   |-- ✅ pointer.js
|   |   |   |   |   |-- ✅ pointerlock.js
|   |   |   |   |   |-- ✅ portals.js
|   |   |   |   |   |-- ✅ prefers-color-scheme.js
|   |   |   |   |   |-- ✅ prefers-reduced-motion.js
|   |   |   |   |   |-- ✅ progress.js
|   |   |   |   |   |-- ✅ promise-finally.js
|   |   |   |   |   |-- ✅ promises.js
|   |   |   |   |   |-- ✅ proximity.js
|   |   |   |   |   |-- ✅ proxy.js
|   |   |   |   |   |-- ✅ publickeypinning.js
|   |   |   |   |   |-- ✅ push-api.js
|   |   |   |   |   |-- ✅ queryselector.js
|   |   |   |   |   |-- ✅ readonly-attr.js
|   |   |   |   |   |-- ✅ referrer-policy.js
|   |   |   |   |   |-- ✅ registerprotocolhandler.js
|   |   |   |   |   |-- ✅ rel-noopener.js
|   |   |   |   |   |-- ✅ rel-noreferrer.js
|   |   |   |   |   |-- ✅ rellist.js
|   |   |   |   |   |-- ✅ rem.js
|   |   |   |   |   |-- ✅ requestanimationframe.js
|   |   |   |   |   |-- ✅ requestidlecallback.js
|   |   |   |   |   |-- ✅ resizeobserver.js
|   |   |   |   |   |-- ✅ resource-timing.js
|   |   |   |   |   |-- ✅ rest-parameters.js
|   |   |   |   |   |-- ✅ rtcpeerconnection.js
|   |   |   |   |   |-- ✅ ruby.js
|   |   |   |   |   |-- ✅ run-in.js
|   |   |   |   |   |-- ✅ same-site-cookie-attribute.js
|   |   |   |   |   |-- ✅ screen-orientation.js
|   |   |   |   |   |-- ✅ script-async.js
|   |   |   |   |   |-- ✅ script-defer.js
|   |   |   |   |   |-- ✅ scrollintoview.js
|   |   |   |   |   |-- ✅ scrollintoviewifneeded.js
|   |   |   |   |   |-- ✅ sdch.js
|   |   |   |   |   |-- ✅ selection-api.js
|   |   |   |   |   |-- ✅ selectlist.js
|   |   |   |   |   |-- ✅ server-timing.js
|   |   |   |   |   |-- ✅ serviceworkers.js
|   |   |   |   |   |-- ✅ setimmediate.js
|   |   |   |   |   |-- ✅ shadowdom.js
|   |   |   |   |   |-- ✅ shadowdomv1.js
|   |   |   |   |   |-- ✅ sharedarraybuffer.js
|   |   |   |   |   |-- ✅ sharedworkers.js
|   |   |   |   |   |-- ✅ sni.js
|   |   |   |   |   |-- ✅ spdy.js
|   |   |   |   |   |-- ✅ speech-recognition.js
|   |   |   |   |   |-- ✅ speech-synthesis.js
|   |   |   |   |   |-- ✅ spellcheck-attribute.js
|   |   |   |   |   |-- ✅ sql-storage.js
|   |   |   |   |   |-- ✅ srcset.js
|   |   |   |   |   |-- ✅ stream.js
|   |   |   |   |   |-- ✅ streams.js
|   |   |   |   |   |-- ✅ stricttransportsecurity.js
|   |   |   |   |   |-- ✅ style-scoped.js
|   |   |   |   |   |-- ✅ subresource-bundling.js
|   |   |   |   |   |-- ✅ subresource-integrity.js
|   |   |   |   |   |-- ✅ svg-css.js
|   |   |   |   |   |-- ✅ svg-filters.js
|   |   |   |   |   |-- ✅ svg-fonts.js
|   |   |   |   |   |-- ✅ svg-fragment.js
|   |   |   |   |   |-- ✅ svg-html.js
|   |   |   |   |   |-- ✅ svg-html5.js
|   |   |   |   |   |-- ✅ svg-img.js
|   |   |   |   |   |-- ✅ svg-smil.js
|   |   |   |   |   |-- ✅ svg.js
|   |   |   |   |   |-- ✅ sxg.js
|   |   |   |   |   |-- ✅ tabindex-attr.js
|   |   |   |   |   |-- ✅ template-literals.js
|   |   |   |   |   |-- ✅ template.js
|   |   |   |   |   |-- ✅ temporal.js
|   |   |   |   |   |-- ✅ testfeat.js
|   |   |   |   |   |-- ✅ text-decoration.js
|   |   |   |   |   |-- ✅ text-emphasis.js
|   |   |   |   |   |-- ✅ text-overflow.js
|   |   |   |   |   |-- ✅ text-size-adjust.js
|   |   |   |   |   |-- ✅ text-stroke.js
|   |   |   |   |   |-- ✅ textcontent.js
|   |   |   |   |   |-- ✅ textencoder.js
|   |   |   |   |   |-- ✅ tls1-1.js
|   |   |   |   |   |-- ✅ tls1-2.js
|   |   |   |   |   |-- ✅ tls1-3.js
|   |   |   |   |   |-- ✅ touch.js
|   |   |   |   |   |-- ✅ transforms2d.js
|   |   |   |   |   |-- ✅ transforms3d.js
|   |   |   |   |   |-- ✅ trusted-types.js
|   |   |   |   |   |-- ✅ ttf.js
|   |   |   |   |   |-- ✅ typedarrays.js
|   |   |   |   |   |-- ✅ u2f.js
|   |   |   |   |   |-- ✅ unhandledrejection.js
|   |   |   |   |   |-- ✅ upgradeinsecurerequests.js
|   |   |   |   |   |-- ✅ url-scroll-to-text-fragment.js
|   |   |   |   |   |-- ✅ url.js
|   |   |   |   |   |-- ✅ urlsearchparams.js
|   |   |   |   |   |-- ✅ use-strict.js
|   |   |   |   |   |-- ✅ user-select-none.js
|   |   |   |   |   |-- ✅ user-timing.js
|   |   |   |   |   |-- ✅ variable-fonts.js
|   |   |   |   |   |-- ✅ vector-effect.js
|   |   |   |   |   |-- ✅ vibration.js
|   |   |   |   |   |-- ✅ video.js
|   |   |   |   |   |-- ✅ videotracks.js
|   |   |   |   |   |-- ✅ view-transitions.js
|   |   |   |   |   |-- ✅ viewport-unit-variants.js
|   |   |   |   |   |-- ✅ viewport-units.js
|   |   |   |   |   |-- ✅ wai-aria.js
|   |   |   |   |   |-- ✅ wake-lock.js
|   |   |   |   |   |-- ✅ wasm-bigint.js
|   |   |   |   |   |-- ✅ wasm-bulk-memory.js
|   |   |   |   |   |-- ✅ wasm-extended-const.js
|   |   |   |   |   |-- ✅ wasm-gc.js
|   |   |   |   |   |-- ✅ wasm-multi-memory.js
|   |   |   |   |   |-- ✅ wasm-multi-value.js
|   |   |   |   |   |-- ✅ wasm-mutable-globals.js
|   |   |   |   |   |-- ✅ wasm-nontrapping-fptoint.js
|   |   |   |   |   |-- ✅ wasm-reference-types.js
|   |   |   |   |   |-- ✅ wasm-relaxed-simd.js
|   |   |   |   |   |-- ✅ wasm-signext.js
|   |   |   |   |   |-- ✅ wasm-simd.js
|   |   |   |   |   |-- ✅ wasm-tail-calls.js
|   |   |   |   |   |-- ✅ wasm-threads.js
|   |   |   |   |   |-- ✅ wasm.js
|   |   |   |   |   |-- ✅ wav.js
|   |   |   |   |   |-- ✅ wbr-element.js
|   |   |   |   |   |-- ✅ web-animation.js
|   |   |   |   |   |-- ✅ web-app-manifest.js
|   |   |   |   |   |-- ✅ web-bluetooth.js
|   |   |   |   |   |-- ✅ web-serial.js
|   |   |   |   |   |-- ✅ web-share.js
|   |   |   |   |   |-- ✅ webauthn.js
|   |   |   |   |   |-- ✅ webcodecs.js
|   |   |   |   |   |-- ✅ webgl.js
|   |   |   |   |   |-- ✅ webgl2.js
|   |   |   |   |   |-- ✅ webgpu.js
|   |   |   |   |   |-- ✅ webhid.js
|   |   |   |   |   |-- ✅ webkit-user-drag.js
|   |   |   |   |   |-- ✅ webm.js
|   |   |   |   |   |-- ✅ webnfc.js
|   |   |   |   |   |-- ✅ webp.js
|   |   |   |   |   |-- ✅ websockets.js
|   |   |   |   |   |-- ✅ webtransport.js
|   |   |   |   |   |-- ✅ webusb.js
|   |   |   |   |   |-- ✅ webvr.js
|   |   |   |   |   |-- ✅ webvtt.js
|   |   |   |   |   |-- ✅ webworkers.js
|   |   |   |   |   |-- ✅ webxr.js
|   |   |   |   |   |-- ✅ will-change.js
|   |   |   |   |   |-- ✅ woff.js
|   |   |   |   |   |-- ✅ woff2.js
|   |   |   |   |   |-- ✅ word-break.js
|   |   |   |   |   |-- ✅ wordwrap.js
|   |   |   |   |   |-- ✅ x-doc-messaging.js
|   |   |   |   |   |-- ✅ x-frame-options.js
|   |   |   |   |   |-- ✅ xhr2.js
|   |   |   |   |   |-- ✅ xhtml.js
|   |   |   |   |   |-- ✅ xhtmlsmil.js
|   |   |   |   |   |-- ✅ xml-serializer.js
|   |   |   |   |   \-- ✅ zstd.js
|   |   |   |   |-- ✅ regions/
|   |   |   |   |   |-- ✅ AD.js
|   |   |   |   |   |-- ✅ AE.js
|   |   |   |   |   |-- ✅ AF.js
|   |   |   |   |   |-- ✅ AG.js
|   |   |   |   |   |-- ✅ AI.js
|   |   |   |   |   |-- ✅ AL.js
|   |   |   |   |   |-- ✅ alt-af.js
|   |   |   |   |   |-- ✅ alt-an.js
|   |   |   |   |   |-- ✅ alt-as.js
|   |   |   |   |   |-- ✅ alt-eu.js
|   |   |   |   |   |-- ✅ alt-na.js
|   |   |   |   |   |-- ✅ alt-oc.js
|   |   |   |   |   |-- ✅ alt-sa.js
|   |   |   |   |   |-- ✅ alt-ww.js
|   |   |   |   |   |-- ✅ AM.js
|   |   |   |   |   |-- ✅ AO.js
|   |   |   |   |   |-- ✅ AR.js
|   |   |   |   |   |-- ✅ AS.js
|   |   |   |   |   |-- ✅ AT.js
|   |   |   |   |   |-- ✅ AU.js
|   |   |   |   |   |-- ✅ AW.js
|   |   |   |   |   |-- ✅ AX.js
|   |   |   |   |   |-- ✅ AZ.js
|   |   |   |   |   |-- ✅ BA.js
|   |   |   |   |   |-- ✅ BB.js
|   |   |   |   |   |-- ✅ BD.js
|   |   |   |   |   |-- ✅ BE.js
|   |   |   |   |   |-- ✅ BF.js
|   |   |   |   |   |-- ✅ BG.js
|   |   |   |   |   |-- ✅ BH.js
|   |   |   |   |   |-- ✅ BI.js
|   |   |   |   |   |-- ✅ BJ.js
|   |   |   |   |   |-- ✅ BM.js
|   |   |   |   |   |-- ✅ BN.js
|   |   |   |   |   |-- ✅ BO.js
|   |   |   |   |   |-- ✅ BR.js
|   |   |   |   |   |-- ✅ BS.js
|   |   |   |   |   |-- ✅ BT.js
|   |   |   |   |   |-- ✅ BW.js
|   |   |   |   |   |-- ✅ BY.js
|   |   |   |   |   |-- ✅ BZ.js
|   |   |   |   |   |-- ✅ CA.js
|   |   |   |   |   |-- ✅ CD.js
|   |   |   |   |   |-- ✅ CF.js
|   |   |   |   |   |-- ✅ CG.js
|   |   |   |   |   |-- ✅ CH.js
|   |   |   |   |   |-- ✅ CI.js
|   |   |   |   |   |-- ✅ CK.js
|   |   |   |   |   |-- ✅ CL.js
|   |   |   |   |   |-- ✅ CM.js
|   |   |   |   |   |-- ✅ CN.js
|   |   |   |   |   |-- ✅ CO.js
|   |   |   |   |   |-- ✅ CR.js
|   |   |   |   |   |-- ✅ CU.js
|   |   |   |   |   |-- ✅ CV.js
|   |   |   |   |   |-- ✅ CX.js
|   |   |   |   |   |-- ✅ CY.js
|   |   |   |   |   |-- ✅ CZ.js
|   |   |   |   |   |-- ✅ DE.js
|   |   |   |   |   |-- ✅ DJ.js
|   |   |   |   |   |-- ✅ DK.js
|   |   |   |   |   |-- ✅ DM.js
|   |   |   |   |   |-- ✅ DO.js
|   |   |   |   |   |-- ✅ DZ.js
|   |   |   |   |   |-- ✅ EC.js
|   |   |   |   |   |-- ✅ EE.js
|   |   |   |   |   |-- ✅ EG.js
|   |   |   |   |   |-- ✅ ER.js
|   |   |   |   |   |-- ✅ ES.js
|   |   |   |   |   |-- ✅ ET.js
|   |   |   |   |   |-- ✅ FI.js
|   |   |   |   |   |-- ✅ FJ.js
|   |   |   |   |   |-- ✅ FK.js
|   |   |   |   |   |-- ✅ FM.js
|   |   |   |   |   |-- ✅ FO.js
|   |   |   |   |   |-- ✅ FR.js
|   |   |   |   |   |-- ✅ GA.js
|   |   |   |   |   |-- ✅ GB.js
|   |   |   |   |   |-- ✅ GD.js
|   |   |   |   |   |-- ✅ GE.js
|   |   |   |   |   |-- ✅ GF.js
|   |   |   |   |   |-- ✅ GG.js
|   |   |   |   |   |-- ✅ GH.js
|   |   |   |   |   |-- ✅ GI.js
|   |   |   |   |   |-- ✅ GL.js
|   |   |   |   |   |-- ✅ GM.js
|   |   |   |   |   |-- ✅ GN.js
|   |   |   |   |   |-- ✅ GP.js
|   |   |   |   |   |-- ✅ GQ.js
|   |   |   |   |   |-- ✅ GR.js
|   |   |   |   |   |-- ✅ GT.js
|   |   |   |   |   |-- ✅ GU.js
|   |   |   |   |   |-- ✅ GW.js
|   |   |   |   |   |-- ✅ GY.js
|   |   |   |   |   |-- ✅ HK.js
|   |   |   |   |   |-- ✅ HN.js
|   |   |   |   |   |-- ✅ HR.js
|   |   |   |   |   |-- ✅ HT.js
|   |   |   |   |   |-- ✅ HU.js
|   |   |   |   |   |-- ✅ ID.js
|   |   |   |   |   |-- ✅ IE.js
|   |   |   |   |   |-- ✅ IL.js
|   |   |   |   |   |-- ✅ IM.js
|   |   |   |   |   |-- ✅ IN.js
|   |   |   |   |   |-- ✅ IQ.js
|   |   |   |   |   |-- ✅ IR.js
|   |   |   |   |   |-- ✅ IS.js
|   |   |   |   |   |-- ✅ IT.js
|   |   |   |   |   |-- ✅ JE.js
|   |   |   |   |   |-- ✅ JM.js
|   |   |   |   |   |-- ✅ JO.js
|   |   |   |   |   |-- ✅ JP.js
|   |   |   |   |   |-- ✅ KE.js
|   |   |   |   |   |-- ✅ KG.js
|   |   |   |   |   |-- ✅ KH.js
|   |   |   |   |   |-- ✅ KI.js
|   |   |   |   |   |-- ✅ KM.js
|   |   |   |   |   |-- ✅ KN.js
|   |   |   |   |   |-- ✅ KP.js
|   |   |   |   |   |-- ✅ KR.js
|   |   |   |   |   |-- ✅ KW.js
|   |   |   |   |   |-- ✅ KY.js
|   |   |   |   |   |-- ✅ KZ.js
|   |   |   |   |   |-- ✅ LA.js
|   |   |   |   |   |-- ✅ LB.js
|   |   |   |   |   |-- ✅ LC.js
|   |   |   |   |   |-- ✅ LI.js
|   |   |   |   |   |-- ✅ LK.js
|   |   |   |   |   |-- ✅ LR.js
|   |   |   |   |   |-- ✅ LS.js
|   |   |   |   |   |-- ✅ LT.js
|   |   |   |   |   |-- ✅ LU.js
|   |   |   |   |   |-- ✅ LV.js
|   |   |   |   |   |-- ✅ LY.js
|   |   |   |   |   |-- ✅ MA.js
|   |   |   |   |   |-- ✅ MC.js
|   |   |   |   |   |-- ✅ MD.js
|   |   |   |   |   |-- ✅ ME.js
|   |   |   |   |   |-- ✅ MG.js
|   |   |   |   |   |-- ✅ MH.js
|   |   |   |   |   |-- ✅ MK.js
|   |   |   |   |   |-- ✅ ML.js
|   |   |   |   |   |-- ✅ MM.js
|   |   |   |   |   |-- ✅ MN.js
|   |   |   |   |   |-- ✅ MO.js
|   |   |   |   |   |-- ✅ MP.js
|   |   |   |   |   |-- ✅ MQ.js
|   |   |   |   |   |-- ✅ MR.js
|   |   |   |   |   |-- ✅ MS.js
|   |   |   |   |   |-- ✅ MT.js
|   |   |   |   |   |-- ✅ MU.js
|   |   |   |   |   |-- ✅ MV.js
|   |   |   |   |   |-- ✅ MW.js
|   |   |   |   |   |-- ✅ MX.js
|   |   |   |   |   |-- ✅ MY.js
|   |   |   |   |   |-- ✅ MZ.js
|   |   |   |   |   |-- ✅ NA.js
|   |   |   |   |   |-- ✅ NC.js
|   |   |   |   |   |-- ✅ NE.js
|   |   |   |   |   |-- ✅ NF.js
|   |   |   |   |   |-- ✅ NG.js
|   |   |   |   |   |-- ✅ NI.js
|   |   |   |   |   |-- ✅ NL.js
|   |   |   |   |   |-- ✅ NO.js
|   |   |   |   |   |-- ✅ NP.js
|   |   |   |   |   |-- ✅ NR.js
|   |   |   |   |   |-- ✅ NU.js
|   |   |   |   |   |-- ✅ NZ.js
|   |   |   |   |   |-- ✅ OM.js
|   |   |   |   |   |-- ✅ PA.js
|   |   |   |   |   |-- ✅ PE.js
|   |   |   |   |   |-- ✅ PF.js
|   |   |   |   |   |-- ✅ PG.js
|   |   |   |   |   |-- ✅ PH.js
|   |   |   |   |   |-- ✅ PK.js
|   |   |   |   |   |-- ✅ PL.js
|   |   |   |   |   |-- ✅ PM.js
|   |   |   |   |   |-- ✅ PN.js
|   |   |   |   |   |-- ✅ PR.js
|   |   |   |   |   |-- ✅ PS.js
|   |   |   |   |   |-- ✅ PT.js
|   |   |   |   |   |-- ✅ PW.js
|   |   |   |   |   |-- ✅ PY.js
|   |   |   |   |   |-- ✅ QA.js
|   |   |   |   |   |-- ✅ RE.js
|   |   |   |   |   |-- ✅ RO.js
|   |   |   |   |   |-- ✅ RS.js
|   |   |   |   |   |-- ✅ RU.js
|   |   |   |   |   |-- ✅ RW.js
|   |   |   |   |   |-- ✅ SA.js
|   |   |   |   |   |-- ✅ SB.js
|   |   |   |   |   |-- ✅ SC.js
|   |   |   |   |   |-- ✅ SD.js
|   |   |   |   |   |-- ✅ SE.js
|   |   |   |   |   |-- ✅ SG.js
|   |   |   |   |   |-- ✅ SH.js
|   |   |   |   |   |-- ✅ SI.js
|   |   |   |   |   |-- ✅ SK.js
|   |   |   |   |   |-- ✅ SL.js
|   |   |   |   |   |-- ✅ SM.js
|   |   |   |   |   |-- ✅ SN.js
|   |   |   |   |   |-- ✅ SO.js
|   |   |   |   |   |-- ✅ SR.js
|   |   |   |   |   |-- ✅ ST.js
|   |   |   |   |   |-- ✅ SV.js
|   |   |   |   |   |-- ✅ SY.js
|   |   |   |   |   |-- ✅ SZ.js
|   |   |   |   |   |-- ✅ TC.js
|   |   |   |   |   |-- ✅ TD.js
|   |   |   |   |   |-- ✅ TG.js
|   |   |   |   |   |-- ✅ TH.js
|   |   |   |   |   |-- ✅ TJ.js
|   |   |   |   |   |-- ✅ TL.js
|   |   |   |   |   |-- ✅ TM.js
|   |   |   |   |   |-- ✅ TN.js
|   |   |   |   |   |-- ✅ TO.js
|   |   |   |   |   |-- ✅ TR.js
|   |   |   |   |   |-- ✅ TT.js
|   |   |   |   |   |-- ✅ TV.js
|   |   |   |   |   |-- ✅ TW.js
|   |   |   |   |   |-- ✅ TZ.js
|   |   |   |   |   |-- ✅ UA.js
|   |   |   |   |   |-- ✅ UG.js
|   |   |   |   |   |-- ✅ US.js
|   |   |   |   |   |-- ✅ UY.js
|   |   |   |   |   |-- ✅ UZ.js
|   |   |   |   |   |-- ✅ VA.js
|   |   |   |   |   |-- ✅ VC.js
|   |   |   |   |   |-- ✅ VE.js
|   |   |   |   |   |-- ✅ VG.js
|   |   |   |   |   |-- ✅ VI.js
|   |   |   |   |   |-- ✅ VN.js
|   |   |   |   |   |-- ✅ VU.js
|   |   |   |   |   |-- ✅ WF.js
|   |   |   |   |   |-- ✅ WS.js
|   |   |   |   |   |-- ✅ YE.js
|   |   |   |   |   |-- ✅ YT.js
|   |   |   |   |   |-- ✅ ZA.js
|   |   |   |   |   |-- ✅ ZM.js
|   |   |   |   |   \-- ✅ ZW.js
|   |   |   |   |-- ✅ agents.js
|   |   |   |   |-- ✅ browsers.js
|   |   |   |   |-- ✅ browserVersions.js
|   |   |   |   \-- ✅ features.js
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ statuses.js
|   |   |   |   |   \-- ✅ supported.js
|   |   |   |   \-- ✅ unpacker/
|   |   |   |       |-- ✅ agents.js
|   |   |   |       |-- ✅ browsers.js
|   |   |   |       |-- ✅ browserVersions.js
|   |   |   |       |-- ✅ feature.js
|   |   |   |       |-- ✅ features.js
|   |   |   |       |-- ✅ index.js
|   |   |   |       \-- ✅ region.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ chai/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ chai/
|   |   |   |   |   |-- ✅ core/
|   |   |   |   |   |   \-- ✅ assertions.js
|   |   |   |   |   |-- ✅ interface/
|   |   |   |   |   |   |-- ✅ assert.js
|   |   |   |   |   |   |-- ✅ expect.js
|   |   |   |   |   |   \-- ✅ should.js
|   |   |   |   |   |-- ✅ utils/
|   |   |   |   |   |   |-- ✅ addChainableMethod.js
|   |   |   |   |   |   |-- ✅ addLengthGuard.js
|   |   |   |   |   |   |-- ✅ addMethod.js
|   |   |   |   |   |   |-- ✅ addProperty.js
|   |   |   |   |   |   |-- ✅ compareByInspect.js
|   |   |   |   |   |   |-- ✅ expectTypes.js
|   |   |   |   |   |   |-- ✅ flag.js
|   |   |   |   |   |   |-- ✅ getActual.js
|   |   |   |   |   |   |-- ✅ getMessage.js
|   |   |   |   |   |   |-- ✅ getOperator.js
|   |   |   |   |   |   |-- ✅ getOwnEnumerableProperties.js
|   |   |   |   |   |   |-- ✅ getOwnEnumerablePropertySymbols.js
|   |   |   |   |   |   |-- ✅ getProperties.js
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ inspect.js
|   |   |   |   |   |   |-- ✅ isNaN.js
|   |   |   |   |   |   |-- ✅ isProxyEnabled.js
|   |   |   |   |   |   |-- ✅ objDisplay.js
|   |   |   |   |   |   |-- ✅ overwriteChainableMethod.js
|   |   |   |   |   |   |-- ✅ overwriteMethod.js
|   |   |   |   |   |   |-- ✅ overwriteProperty.js
|   |   |   |   |   |   |-- ✅ proxify.js
|   |   |   |   |   |   |-- ✅ test.js
|   |   |   |   |   |   |-- ✅ transferFlags.js
|   |   |   |   |   |   \-- ✅ type-detect.js
|   |   |   |   |   |-- ✅ assertion.js
|   |   |   |   |   \-- ✅ config.js
|   |   |   |   \-- ✅ chai.js
|   |   |   |-- ✅ chai.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ register-assert.js
|   |   |   |-- ✅ register-expect.js
|   |   |   \-- ✅ register-should.js
|   |   |-- ✅ chalk/
|   |   |   |-- ✅ source/
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ templates.js
|   |   |   |   \-- ✅ util.js
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ chart.js/
|   |   |   |-- ✅ auto/
|   |   |   |   |-- ✅ auto.cjs
|   |   |   |   |-- ✅ auto.d.ts
|   |   |   |   |-- ✅ auto.js
|   |   |   |   \-- ✅ package.json
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ chunks/
|   |   |   |   |   |-- ✅ helpers.dataset.cjs
|   |   |   |   |   |-- ✅ helpers.dataset.cjs.map
|   |   |   |   |   |-- ✅ helpers.dataset.js
|   |   |   |   |   \-- ✅ helpers.dataset.js.map
|   |   |   |   |-- ✅ controllers/
|   |   |   |   |   |-- ✅ controller.bar.d.ts
|   |   |   |   |   |-- ✅ controller.bubble.d.ts
|   |   |   |   |   |-- ✅ controller.doughnut.d.ts
|   |   |   |   |   |-- ✅ controller.line.d.ts
|   |   |   |   |   |-- ✅ controller.pie.d.ts
|   |   |   |   |   |-- ✅ controller.polarArea.d.ts
|   |   |   |   |   |-- ✅ controller.radar.d.ts
|   |   |   |   |   |-- ✅ controller.scatter.d.ts
|   |   |   |   |   \-- ✅ index.d.ts
|   |   |   |   |-- ✅ core/
|   |   |   |   |   |-- ✅ core.adapters.d.ts
|   |   |   |   |   |-- ✅ core.animation.d.ts
|   |   |   |   |   |-- ✅ core.animations.d.ts
|   |   |   |   |   |-- ✅ core.animations.defaults.d.ts
|   |   |   |   |   |-- ✅ core.animator.d.ts
|   |   |   |   |   |-- ✅ core.config.d.ts
|   |   |   |   |   |-- ✅ core.controller.d.ts
|   |   |   |   |   |-- ✅ core.datasetController.d.ts
|   |   |   |   |   |-- ✅ core.defaults.d.ts
|   |   |   |   |   |-- ✅ core.element.d.ts
|   |   |   |   |   |-- ✅ core.interaction.d.ts
|   |   |   |   |   |-- ✅ core.layouts.d.ts
|   |   |   |   |   |-- ✅ core.layouts.defaults.d.ts
|   |   |   |   |   |-- ✅ core.plugins.d.ts
|   |   |   |   |   |-- ✅ core.registry.d.ts
|   |   |   |   |   |-- ✅ core.scale.autoskip.d.ts
|   |   |   |   |   |-- ✅ core.scale.d.ts
|   |   |   |   |   |-- ✅ core.scale.defaults.d.ts
|   |   |   |   |   |-- ✅ core.ticks.d.ts
|   |   |   |   |   |-- ✅ core.typedRegistry.d.ts
|   |   |   |   |   \-- ✅ index.d.ts
|   |   |   |   |-- ✅ elements/
|   |   |   |   |   |-- ✅ element.arc.d.ts
|   |   |   |   |   |-- ✅ element.bar.d.ts
|   |   |   |   |   |-- ✅ element.line.d.ts
|   |   |   |   |   |-- ✅ element.point.d.ts
|   |   |   |   |   \-- ✅ index.d.ts
|   |   |   |   |-- ✅ helpers/
|   |   |   |   |   |-- ✅ helpers.canvas.d.ts
|   |   |   |   |   |-- ✅ helpers.collection.d.ts
|   |   |   |   |   |-- ✅ helpers.color.d.ts
|   |   |   |   |   |-- ✅ helpers.config.d.ts
|   |   |   |   |   |-- ✅ helpers.config.types.d.ts
|   |   |   |   |   |-- ✅ helpers.core.d.ts
|   |   |   |   |   |-- ✅ helpers.curve.d.ts
|   |   |   |   |   |-- ✅ helpers.dataset.d.ts
|   |   |   |   |   |-- ✅ helpers.dom.d.ts
|   |   |   |   |   |-- ✅ helpers.easing.d.ts
|   |   |   |   |   |-- ✅ helpers.extras.d.ts
|   |   |   |   |   |-- ✅ helpers.interpolation.d.ts
|   |   |   |   |   |-- ✅ helpers.intl.d.ts
|   |   |   |   |   |-- ✅ helpers.math.d.ts
|   |   |   |   |   |-- ✅ helpers.options.d.ts
|   |   |   |   |   |-- ✅ helpers.rtl.d.ts
|   |   |   |   |   |-- ✅ helpers.segment.d.ts
|   |   |   |   |   \-- ✅ index.d.ts
|   |   |   |   |-- ✅ platform/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ platform.base.d.ts
|   |   |   |   |   |-- ✅ platform.basic.d.ts
|   |   |   |   |   \-- ✅ platform.dom.d.ts
|   |   |   |   |-- ✅ plugins/
|   |   |   |   |   |-- ✅ plugin.filler/
|   |   |   |   |   |   |-- ✅ filler.drawing.d.ts
|   |   |   |   |   |   |-- ✅ filler.helper.d.ts
|   |   |   |   |   |   |-- ✅ filler.options.d.ts
|   |   |   |   |   |   |-- ✅ filler.segment.d.ts
|   |   |   |   |   |   |-- ✅ filler.target.d.ts
|   |   |   |   |   |   |-- ✅ filler.target.stack.d.ts
|   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   \-- ✅ simpleArc.d.ts
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ plugin.colors.d.ts
|   |   |   |   |   |-- ✅ plugin.decimation.d.ts
|   |   |   |   |   |-- ✅ plugin.legend.d.ts
|   |   |   |   |   |-- ✅ plugin.subtitle.d.ts
|   |   |   |   |   |-- ✅ plugin.title.d.ts
|   |   |   |   |   \-- ✅ plugin.tooltip.d.ts
|   |   |   |   |-- ✅ scales/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ scale.category.d.ts
|   |   |   |   |   |-- ✅ scale.linear.d.ts
|   |   |   |   |   |-- ✅ scale.linearbase.d.ts
|   |   |   |   |   |-- ✅ scale.logarithmic.d.ts
|   |   |   |   |   |-- ✅ scale.radialLinear.d.ts
|   |   |   |   |   |-- ✅ scale.time.d.ts
|   |   |   |   |   \-- ✅ scale.timeseries.d.ts
|   |   |   |   |-- ✅ types/
|   |   |   |   |   |-- ✅ animation.d.ts
|   |   |   |   |   |-- ✅ basic.d.ts
|   |   |   |   |   |-- ✅ color.d.ts
|   |   |   |   |   |-- ✅ geometric.d.ts
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ layout.d.ts
|   |   |   |   |   \-- ✅ utils.d.ts
|   |   |   |   |-- ✅ chart.cjs
|   |   |   |   |-- ✅ chart.cjs.map
|   |   |   |   |-- ✅ chart.js
|   |   |   |   |-- ✅ chart.js.map
|   |   |   |   |-- ✅ chart.umd.js
|   |   |   |   |-- ✅ chart.umd.js.map
|   |   |   |   |-- ✅ chart.umd.min.js
|   |   |   |   |-- ✅ chart.umd.min.js.map
|   |   |   |   |-- ✅ helpers.cjs
|   |   |   |   |-- ✅ helpers.cjs.map
|   |   |   |   |-- ✅ helpers.js
|   |   |   |   |-- ✅ helpers.js.map
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.umd.d.ts
|   |   |   |   \-- ✅ types.d.ts
|   |   |   |-- ✅ helpers/
|   |   |   |   |-- ✅ helpers.cjs
|   |   |   |   |-- ✅ helpers.d.ts
|   |   |   |   |-- ✅ helpers.js
|   |   |   |   \-- ✅ package.json
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ check-error/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ chokidar/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ constants.js
|   |   |   |   |-- ✅ fsevents-handler.js
|   |   |   |   \-- ✅ nodefs-handler.js
|   |   |   |-- ✅ node_modules/
|   |   |   |   \-- ✅ glob-parent/
|   |   |   |       |-- ✅ CHANGELOG.md
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ LICENSE
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ README.md
|   |   |   |-- ✅ types/
|   |   |   |   \-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ ci-info/
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ vendors.json
|   |   |-- ✅ color-convert/
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ conversions.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ route.js
|   |   |-- ✅ color-name/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ combined-stream/
|   |   |   |-- ✅ lib/
|   |   |   |   \-- ✅ combined_stream.js
|   |   |   |-- ✅ License
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ Readme.md
|   |   |   \-- ✅ yarn.lock
|   |   |-- ✅ commander/
|   |   |   |-- ✅ typings/
|   |   |   |   \-- ✅ index.d.ts
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ Readme.md
|   |   |-- ✅ convert-source-map/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ cross-spawn/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ util/
|   |   |   |   |   |-- ✅ escape.js
|   |   |   |   |   |-- ✅ readShebang.js
|   |   |   |   |   \-- ✅ resolveCommand.js
|   |   |   |   |-- ✅ enoent.js
|   |   |   |   \-- ✅ parse.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ css.escape/
|   |   |   |-- ✅ css.escape.js
|   |   |   |-- ✅ LICENSE-MIT.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ cssesc/
|   |   |   |-- ✅ bin/
|   |   |   |   \-- ✅ cssesc
|   |   |   |-- ✅ man/
|   |   |   |   \-- ✅ cssesc.1
|   |   |   |-- ✅ cssesc.js
|   |   |   |-- ✅ LICENSE-MIT.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ cssstyle/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ properties/
|   |   |   |   |   |-- ✅ azimuth.js
|   |   |   |   |   |-- ✅ background.js
|   |   |   |   |   |-- ✅ backgroundAttachment.js
|   |   |   |   |   |-- ✅ backgroundColor.js
|   |   |   |   |   |-- ✅ backgroundImage.js
|   |   |   |   |   |-- ✅ backgroundPosition.js
|   |   |   |   |   |-- ✅ backgroundRepeat.js
|   |   |   |   |   |-- ✅ border.js
|   |   |   |   |   |-- ✅ borderBottom.js
|   |   |   |   |   |-- ✅ borderBottomColor.js
|   |   |   |   |   |-- ✅ borderBottomStyle.js
|   |   |   |   |   |-- ✅ borderBottomWidth.js
|   |   |   |   |   |-- ✅ borderCollapse.js
|   |   |   |   |   |-- ✅ borderColor.js
|   |   |   |   |   |-- ✅ borderLeft.js
|   |   |   |   |   |-- ✅ borderLeftColor.js
|   |   |   |   |   |-- ✅ borderLeftStyle.js
|   |   |   |   |   |-- ✅ borderLeftWidth.js
|   |   |   |   |   |-- ✅ borderRight.js
|   |   |   |   |   |-- ✅ borderRightColor.js
|   |   |   |   |   |-- ✅ borderRightStyle.js
|   |   |   |   |   |-- ✅ borderRightWidth.js
|   |   |   |   |   |-- ✅ borderSpacing.js
|   |   |   |   |   |-- ✅ borderStyle.js
|   |   |   |   |   |-- ✅ borderTop.js
|   |   |   |   |   |-- ✅ borderTopColor.js
|   |   |   |   |   |-- ✅ borderTopStyle.js
|   |   |   |   |   |-- ✅ borderTopWidth.js
|   |   |   |   |   |-- ✅ borderWidth.js
|   |   |   |   |   |-- ✅ bottom.js
|   |   |   |   |   |-- ✅ clear.js
|   |   |   |   |   |-- ✅ clip.js
|   |   |   |   |   |-- ✅ color.js
|   |   |   |   |   |-- ✅ cssFloat.js
|   |   |   |   |   |-- ✅ flex.js
|   |   |   |   |   |-- ✅ flexBasis.js
|   |   |   |   |   |-- ✅ flexGrow.js
|   |   |   |   |   |-- ✅ flexShrink.js
|   |   |   |   |   |-- ✅ float.js
|   |   |   |   |   |-- ✅ floodColor.js
|   |   |   |   |   |-- ✅ font.js
|   |   |   |   |   |-- ✅ fontFamily.js
|   |   |   |   |   |-- ✅ fontSize.js
|   |   |   |   |   |-- ✅ fontStyle.js
|   |   |   |   |   |-- ✅ fontVariant.js
|   |   |   |   |   |-- ✅ fontWeight.js
|   |   |   |   |   |-- ✅ height.js
|   |   |   |   |   |-- ✅ left.js
|   |   |   |   |   |-- ✅ lightingColor.js
|   |   |   |   |   |-- ✅ lineHeight.js
|   |   |   |   |   |-- ✅ margin.js
|   |   |   |   |   |-- ✅ marginBottom.js
|   |   |   |   |   |-- ✅ marginLeft.js
|   |   |   |   |   |-- ✅ marginRight.js
|   |   |   |   |   |-- ✅ marginTop.js
|   |   |   |   |   |-- ✅ opacity.js
|   |   |   |   |   |-- ✅ outlineColor.js
|   |   |   |   |   |-- ✅ padding.js
|   |   |   |   |   |-- ✅ paddingBottom.js
|   |   |   |   |   |-- ✅ paddingLeft.js
|   |   |   |   |   |-- ✅ paddingRight.js
|   |   |   |   |   |-- ✅ paddingTop.js
|   |   |   |   |   |-- ✅ right.js
|   |   |   |   |   |-- ✅ stopColor.js
|   |   |   |   |   |-- ✅ textLineThroughColor.js
|   |   |   |   |   |-- ✅ textOverlineColor.js
|   |   |   |   |   |-- ✅ textUnderlineColor.js
|   |   |   |   |   |-- ✅ top.js
|   |   |   |   |   |-- ✅ webkitBorderAfterColor.js
|   |   |   |   |   |-- ✅ webkitBorderBeforeColor.js
|   |   |   |   |   |-- ✅ webkitBorderEndColor.js
|   |   |   |   |   |-- ✅ webkitBorderStartColor.js
|   |   |   |   |   |-- ✅ webkitColumnRuleColor.js
|   |   |   |   |   |-- ✅ webkitMatchNearestMailBlockquoteColor.js
|   |   |   |   |   |-- ✅ webkitTapHighlightColor.js
|   |   |   |   |   |-- ✅ webkitTextEmphasisColor.js
|   |   |   |   |   |-- ✅ webkitTextFillColor.js
|   |   |   |   |   |-- ✅ webkitTextStrokeColor.js
|   |   |   |   |   \-- ✅ width.js
|   |   |   |   |-- ✅ utils/
|   |   |   |   |   |-- ✅ colorSpace.js
|   |   |   |   |   \-- ✅ getBasicPropertyDescriptor.js
|   |   |   |   |-- ✅ allExtraProperties.js
|   |   |   |   |-- ✅ allProperties.js
|   |   |   |   |-- ✅ allWebkitProperties.js
|   |   |   |   |-- ✅ constants.js
|   |   |   |   |-- ✅ CSSStyleDeclaration.js
|   |   |   |   |-- ✅ CSSStyleDeclaration.test.js
|   |   |   |   |-- ✅ implementedProperties.js
|   |   |   |   |-- ✅ named_colors.json
|   |   |   |   |-- ✅ parsers.js
|   |   |   |   |-- ✅ parsers.test.js
|   |   |   |   \-- ✅ properties.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ csstype/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js.flow
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ data-urls/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ parser.js
|   |   |   |   \-- ✅ utils.js
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ debug/
|   |   |   |-- ✅ src/
|   |   |   |   |-- ✅ browser.js
|   |   |   |   |-- ✅ common.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ node.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ decimal.js/
|   |   |   |-- ✅ decimal.d.ts
|   |   |   |-- ✅ decimal.js
|   |   |   |-- ✅ decimal.mjs
|   |   |   |-- ✅ LICENCE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ deep-eql/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ deep-equal/
|   |   |   |-- ✅ example/
|   |   |   |   \-- ✅ cmp.js
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ _tape.js
|   |   |   |   \-- ✅ cmp.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ assert.js
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.markdown
|   |   |-- ✅ define-data-property/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ define-properties/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ delayed-stream/
|   |   |   |-- ✅ lib/
|   |   |   |   \-- ✅ delayed_stream.js
|   |   |   |-- ✅ .npmignore
|   |   |   |-- ✅ License
|   |   |   |-- ✅ Makefile
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ Readme.md
|   |   |-- ✅ didyoumean/
|   |   |   |-- ✅ didYouMean-1.2.1.js
|   |   |   |-- ✅ didYouMean-1.2.1.min.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ diff-sequences/
|   |   |   |-- ✅ build/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ dlv/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ dlv.es.js
|   |   |   |   |-- ✅ dlv.es.js.map
|   |   |   |   |-- ✅ dlv.js
|   |   |   |   |-- ✅ dlv.js.map
|   |   |   |   |-- ✅ dlv.umd.js
|   |   |   |   \-- ✅ dlv.umd.js.map
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ dom-accessibility-api/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ polyfills/
|   |   |   |   |   |-- ✅ array.from.d.ts
|   |   |   |   |   |-- ✅ array.from.d.ts.map
|   |   |   |   |   |-- ✅ array.from.js
|   |   |   |   |   |-- ✅ array.from.js.map
|   |   |   |   |   |-- ✅ array.from.mjs
|   |   |   |   |   |-- ✅ array.from.mjs.map
|   |   |   |   |   |-- ✅ iterator.d.js
|   |   |   |   |   |-- ✅ iterator.d.js.map
|   |   |   |   |   |-- ✅ iterator.d.mjs
|   |   |   |   |   |-- ✅ iterator.d.mjs.map
|   |   |   |   |   |-- ✅ SetLike.d.ts
|   |   |   |   |   |-- ✅ SetLike.d.ts.map
|   |   |   |   |   |-- ✅ SetLike.js
|   |   |   |   |   |-- ✅ SetLike.js.map
|   |   |   |   |   |-- ✅ SetLike.mjs
|   |   |   |   |   \-- ✅ SetLike.mjs.map
|   |   |   |   |-- ✅ accessible-description.d.ts
|   |   |   |   |-- ✅ accessible-description.d.ts.map
|   |   |   |   |-- ✅ accessible-description.js
|   |   |   |   |-- ✅ accessible-description.js.map
|   |   |   |   |-- ✅ accessible-description.mjs
|   |   |   |   |-- ✅ accessible-description.mjs.map
|   |   |   |   |-- ✅ accessible-name-and-description.d.ts
|   |   |   |   |-- ✅ accessible-name-and-description.d.ts.map
|   |   |   |   |-- ✅ accessible-name-and-description.js
|   |   |   |   |-- ✅ accessible-name-and-description.js.map
|   |   |   |   |-- ✅ accessible-name-and-description.mjs
|   |   |   |   |-- ✅ accessible-name-and-description.mjs.map
|   |   |   |   |-- ✅ accessible-name.d.ts
|   |   |   |   |-- ✅ accessible-name.d.ts.map
|   |   |   |   |-- ✅ accessible-name.js
|   |   |   |   |-- ✅ accessible-name.js.map
|   |   |   |   |-- ✅ accessible-name.mjs
|   |   |   |   |-- ✅ accessible-name.mjs.map
|   |   |   |   |-- ✅ getRole.d.ts
|   |   |   |   |-- ✅ getRole.d.ts.map
|   |   |   |   |-- ✅ getRole.js
|   |   |   |   |-- ✅ getRole.js.map
|   |   |   |   |-- ✅ getRole.mjs
|   |   |   |   |-- ✅ getRole.mjs.map
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ index.js.map
|   |   |   |   |-- ✅ index.mjs
|   |   |   |   |-- ✅ index.mjs.map
|   |   |   |   |-- ✅ is-disabled.d.ts
|   |   |   |   |-- ✅ is-disabled.d.ts.map
|   |   |   |   |-- ✅ is-disabled.js
|   |   |   |   |-- ✅ is-disabled.js.map
|   |   |   |   |-- ✅ is-disabled.mjs
|   |   |   |   |-- ✅ is-disabled.mjs.map
|   |   |   |   |-- ✅ is-inaccessible.d.ts
|   |   |   |   |-- ✅ is-inaccessible.d.ts.map
|   |   |   |   |-- ✅ is-inaccessible.js
|   |   |   |   |-- ✅ is-inaccessible.js.map
|   |   |   |   |-- ✅ is-inaccessible.mjs
|   |   |   |   |-- ✅ is-inaccessible.mjs.map
|   |   |   |   |-- ✅ util.d.ts
|   |   |   |   |-- ✅ util.d.ts.map
|   |   |   |   |-- ✅ util.js
|   |   |   |   |-- ✅ util.js.map
|   |   |   |   |-- ✅ util.mjs
|   |   |   |   \-- ✅ util.mjs.map
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ domexception/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ DOMException-impl.js
|   |   |   |   |-- ✅ DOMException.js
|   |   |   |   |-- ✅ Function.js
|   |   |   |   |-- ✅ legacy-error-codes.json
|   |   |   |   |-- ✅ utils.js
|   |   |   |   \-- ✅ VoidFunction.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ webidl2js-wrapper.js
|   |   |-- ✅ dunder-proto/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ get.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ set.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ get.d.ts
|   |   |   |-- ✅ get.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ set.d.ts
|   |   |   |-- ✅ set.js
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ eastasianwidth/
|   |   |   |-- ✅ eastasianwidth.js
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ electron-to-chromium/
|   |   |   |-- ✅ chromium-versions.js
|   |   |   |-- ✅ chromium-versions.json
|   |   |   |-- ✅ full-chromium-versions.js
|   |   |   |-- ✅ full-chromium-versions.json
|   |   |   |-- ✅ full-versions.js
|   |   |   |-- ✅ full-versions.json
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ versions.js
|   |   |   \-- ✅ versions.json
|   |   |-- ✅ emoji-regex/
|   |   |   |-- ✅ es2015/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ RGI_Emoji.d.ts
|   |   |   |   |-- ✅ RGI_Emoji.js
|   |   |   |   |-- ✅ text.d.ts
|   |   |   |   \-- ✅ text.js
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE-MIT.txt
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ RGI_Emoji.d.ts
|   |   |   |-- ✅ RGI_Emoji.js
|   |   |   |-- ✅ text.d.ts
|   |   |   \-- ✅ text.js
|   |   |-- ✅ entities/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ commonjs/
|   |   |   |   |   |-- ✅ generated/
|   |   |   |   |   |   |-- ✅ decode-data-html.d.ts
|   |   |   |   |   |   |-- ✅ decode-data-html.d.ts.map
|   |   |   |   |   |   |-- ✅ decode-data-html.js
|   |   |   |   |   |   |-- ✅ decode-data-html.js.map
|   |   |   |   |   |   |-- ✅ decode-data-xml.d.ts
|   |   |   |   |   |   |-- ✅ decode-data-xml.d.ts.map
|   |   |   |   |   |   |-- ✅ decode-data-xml.js
|   |   |   |   |   |   |-- ✅ decode-data-xml.js.map
|   |   |   |   |   |   |-- ✅ encode-html.d.ts
|   |   |   |   |   |   |-- ✅ encode-html.d.ts.map
|   |   |   |   |   |   |-- ✅ encode-html.js
|   |   |   |   |   |   \-- ✅ encode-html.js.map
|   |   |   |   |   |-- ✅ decode-codepoint.d.ts
|   |   |   |   |   |-- ✅ decode-codepoint.d.ts.map
|   |   |   |   |   |-- ✅ decode-codepoint.js
|   |   |   |   |   |-- ✅ decode-codepoint.js.map
|   |   |   |   |   |-- ✅ decode.d.ts
|   |   |   |   |   |-- ✅ decode.d.ts.map
|   |   |   |   |   |-- ✅ decode.js
|   |   |   |   |   |-- ✅ decode.js.map
|   |   |   |   |   |-- ✅ encode.d.ts
|   |   |   |   |   |-- ✅ encode.d.ts.map
|   |   |   |   |   |-- ✅ encode.js
|   |   |   |   |   |-- ✅ encode.js.map
|   |   |   |   |   |-- ✅ escape.d.ts
|   |   |   |   |   |-- ✅ escape.d.ts.map
|   |   |   |   |   |-- ✅ escape.js
|   |   |   |   |   |-- ✅ escape.js.map
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   \-- ✅ package.json
|   |   |   |   \-- ✅ esm/
|   |   |   |       |-- ✅ generated/
|   |   |   |       |   |-- ✅ decode-data-html.d.ts
|   |   |   |       |   |-- ✅ decode-data-html.d.ts.map
|   |   |   |       |   |-- ✅ decode-data-html.js
|   |   |   |       |   |-- ✅ decode-data-html.js.map
|   |   |   |       |   |-- ✅ decode-data-xml.d.ts
|   |   |   |       |   |-- ✅ decode-data-xml.d.ts.map
|   |   |   |       |   |-- ✅ decode-data-xml.js
|   |   |   |       |   |-- ✅ decode-data-xml.js.map
|   |   |   |       |   |-- ✅ encode-html.d.ts
|   |   |   |       |   |-- ✅ encode-html.d.ts.map
|   |   |   |       |   |-- ✅ encode-html.js
|   |   |   |       |   \-- ✅ encode-html.js.map
|   |   |   |       |-- ✅ decode-codepoint.d.ts
|   |   |   |       |-- ✅ decode-codepoint.d.ts.map
|   |   |   |       |-- ✅ decode-codepoint.js
|   |   |   |       |-- ✅ decode-codepoint.js.map
|   |   |   |       |-- ✅ decode.d.ts
|   |   |   |       |-- ✅ decode.d.ts.map
|   |   |   |       |-- ✅ decode.js
|   |   |   |       |-- ✅ decode.js.map
|   |   |   |       |-- ✅ encode.d.ts
|   |   |   |       |-- ✅ encode.d.ts.map
|   |   |   |       |-- ✅ encode.js
|   |   |   |       |-- ✅ encode.js.map
|   |   |   |       |-- ✅ escape.d.ts
|   |   |   |       |-- ✅ escape.d.ts.map
|   |   |   |       |-- ✅ escape.js
|   |   |   |       |-- ✅ escape.js.map
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.d.ts.map
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ index.js.map
|   |   |   |       \-- ✅ package.json
|   |   |   |-- ✅ src/
|   |   |   |   |-- ✅ generated/
|   |   |   |   |   |-- ✅ .eslintrc.json
|   |   |   |   |   |-- ✅ decode-data-html.ts
|   |   |   |   |   |-- ✅ decode-data-xml.ts
|   |   |   |   |   \-- ✅ encode-html.ts
|   |   |   |   |-- ✅ decode-codepoint.ts
|   |   |   |   |-- ✅ decode.spec.ts
|   |   |   |   |-- ✅ decode.ts
|   |   |   |   |-- ✅ encode.spec.ts
|   |   |   |   |-- ✅ encode.ts
|   |   |   |   |-- ✅ escape.spec.ts
|   |   |   |   |-- ✅ escape.ts
|   |   |   |   |-- ✅ index.spec.ts
|   |   |   |   \-- ✅ index.ts
|   |   |   |-- ✅ decode.d.ts
|   |   |   |-- ✅ decode.js
|   |   |   |-- ✅ escape.d.ts
|   |   |   |-- ✅ escape.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ es-define-property/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ es-errors/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ eval.d.ts
|   |   |   |-- ✅ eval.js
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ range.d.ts
|   |   |   |-- ✅ range.js
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ ref.d.ts
|   |   |   |-- ✅ ref.js
|   |   |   |-- ✅ syntax.d.ts
|   |   |   |-- ✅ syntax.js
|   |   |   |-- ✅ tsconfig.json
|   |   |   |-- ✅ type.d.ts
|   |   |   |-- ✅ type.js
|   |   |   |-- ✅ uri.d.ts
|   |   |   \-- ✅ uri.js
|   |   |-- ✅ es-get-iterator/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ core-js.js
|   |   |   |   |-- ✅ es6-shim.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ node.js
|   |   |   |   \-- ✅ node.mjs
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ node.js
|   |   |   |-- ✅ node.mjs
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ es-module-lexer/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ lexer.asm.js
|   |   |   |   |-- ✅ lexer.cjs
|   |   |   |   \-- ✅ lexer.js
|   |   |   |-- ✅ types/
|   |   |   |   \-- ✅ lexer.d.ts
|   |   |   |-- ✅ lexer.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ es-object-atoms/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ isObject.d.ts
|   |   |   |-- ✅ isObject.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ RequireObjectCoercible.d.ts
|   |   |   |-- ✅ RequireObjectCoercible.js
|   |   |   |-- ✅ ToObject.d.ts
|   |   |   |-- ✅ ToObject.js
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ es-set-tostringtag/
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ esbuild/
|   |   |   |-- ✅ bin/
|   |   |   |   \-- ✅ esbuild
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ main.d.ts
|   |   |   |   \-- ✅ main.js
|   |   |   |-- ✅ install.js
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ escalade/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ index.mjs
|   |   |   |-- ✅ sync/
|   |   |   |   |-- ✅ index.d.mts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ index.mjs
|   |   |   |-- ✅ index.d.mts
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ escape-string-regexp/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ estree-walker/
|   |   |   |-- ✅ src/
|   |   |   |   |-- ✅ async.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ sync.js
|   |   |   |   \-- ✅ walker.js
|   |   |   |-- ✅ types/
|   |   |   |   |-- ✅ async.d.ts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ sync.d.ts
|   |   |   |   \-- ✅ walker.d.ts
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ expect/
|   |   |   |-- ✅ build/
|   |   |   |   |-- ✅ asymmetricMatchers.js
|   |   |   |   |-- ✅ extractExpectedAssertionsErrors.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ jestMatchersObject.js
|   |   |   |   |-- ✅ matchers.js
|   |   |   |   |-- ✅ print.js
|   |   |   |   |-- ✅ spyMatchers.js
|   |   |   |   |-- ✅ toThrowMatchers.js
|   |   |   |   \-- ✅ types.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ expect-type/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ branding.d.ts
|   |   |   |   |-- ✅ branding.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ messages.d.ts
|   |   |   |   |-- ✅ messages.js
|   |   |   |   |-- ✅ overloads.d.ts
|   |   |   |   |-- ✅ overloads.js
|   |   |   |   |-- ✅ utils.d.ts
|   |   |   |   \-- ✅ utils.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ SECURITY.md
|   |   |-- ✅ fast-glob/
|   |   |   |-- ✅ node_modules/
|   |   |   |   \-- ✅ glob-parent/
|   |   |   |       |-- ✅ CHANGELOG.md
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ LICENSE
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ README.md
|   |   |   |-- ✅ out/
|   |   |   |   |-- ✅ managers/
|   |   |   |   |   |-- ✅ tasks.d.ts
|   |   |   |   |   \-- ✅ tasks.js
|   |   |   |   |-- ✅ providers/
|   |   |   |   |   |-- ✅ filters/
|   |   |   |   |   |   |-- ✅ deep.d.ts
|   |   |   |   |   |   |-- ✅ deep.js
|   |   |   |   |   |   |-- ✅ entry.d.ts
|   |   |   |   |   |   |-- ✅ entry.js
|   |   |   |   |   |   |-- ✅ error.d.ts
|   |   |   |   |   |   \-- ✅ error.js
|   |   |   |   |   |-- ✅ matchers/
|   |   |   |   |   |   |-- ✅ matcher.d.ts
|   |   |   |   |   |   |-- ✅ matcher.js
|   |   |   |   |   |   |-- ✅ partial.d.ts
|   |   |   |   |   |   \-- ✅ partial.js
|   |   |   |   |   |-- ✅ transformers/
|   |   |   |   |   |   |-- ✅ entry.d.ts
|   |   |   |   |   |   \-- ✅ entry.js
|   |   |   |   |   |-- ✅ async.d.ts
|   |   |   |   |   |-- ✅ async.js
|   |   |   |   |   |-- ✅ provider.d.ts
|   |   |   |   |   |-- ✅ provider.js
|   |   |   |   |   |-- ✅ stream.d.ts
|   |   |   |   |   |-- ✅ stream.js
|   |   |   |   |   |-- ✅ sync.d.ts
|   |   |   |   |   \-- ✅ sync.js
|   |   |   |   |-- ✅ readers/
|   |   |   |   |   |-- ✅ async.d.ts
|   |   |   |   |   |-- ✅ async.js
|   |   |   |   |   |-- ✅ reader.d.ts
|   |   |   |   |   |-- ✅ reader.js
|   |   |   |   |   |-- ✅ stream.d.ts
|   |   |   |   |   |-- ✅ stream.js
|   |   |   |   |   |-- ✅ sync.d.ts
|   |   |   |   |   \-- ✅ sync.js
|   |   |   |   |-- ✅ types/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ utils/
|   |   |   |   |   |-- ✅ array.d.ts
|   |   |   |   |   |-- ✅ array.js
|   |   |   |   |   |-- ✅ errno.d.ts
|   |   |   |   |   |-- ✅ errno.js
|   |   |   |   |   |-- ✅ fs.d.ts
|   |   |   |   |   |-- ✅ fs.js
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ path.d.ts
|   |   |   |   |   |-- ✅ path.js
|   |   |   |   |   |-- ✅ pattern.d.ts
|   |   |   |   |   |-- ✅ pattern.js
|   |   |   |   |   |-- ✅ stream.d.ts
|   |   |   |   |   |-- ✅ stream.js
|   |   |   |   |   |-- ✅ string.d.ts
|   |   |   |   |   \-- ✅ string.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ settings.d.ts
|   |   |   |   \-- ✅ settings.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ fastq/
|   |   |   |-- ✅ .github/
|   |   |   |   |-- ✅ workflows/
|   |   |   |   |   \-- ✅ ci.yml
|   |   |   |   \-- ✅ dependabot.yml
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ example.ts
|   |   |   |   |-- ✅ promise.js
|   |   |   |   |-- ✅ test.js
|   |   |   |   \-- ✅ tsconfig.json
|   |   |   |-- ✅ bench.js
|   |   |   |-- ✅ example.js
|   |   |   |-- ✅ example.mjs
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ queue.js
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ SECURITY.md
|   |   |-- ✅ fill-range/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ follow-redirects/
|   |   |   |-- ✅ debug.js
|   |   |   |-- ✅ http.js
|   |   |   |-- ✅ https.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ for-each/
|   |   |   |-- ✅ .github/
|   |   |   |   |-- ✅ FUNDING.yml
|   |   |   |   \-- ✅ SECURITY.md
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ test.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ foreground-child/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ commonjs/
|   |   |   |   |   |-- ✅ all-signals.d.ts
|   |   |   |   |   |-- ✅ all-signals.d.ts.map
|   |   |   |   |   |-- ✅ all-signals.js
|   |   |   |   |   |-- ✅ all-signals.js.map
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |-- ✅ proxy-signals.d.ts
|   |   |   |   |   |-- ✅ proxy-signals.d.ts.map
|   |   |   |   |   |-- ✅ proxy-signals.js
|   |   |   |   |   |-- ✅ proxy-signals.js.map
|   |   |   |   |   |-- ✅ watchdog.d.ts
|   |   |   |   |   |-- ✅ watchdog.d.ts.map
|   |   |   |   |   |-- ✅ watchdog.js
|   |   |   |   |   \-- ✅ watchdog.js.map
|   |   |   |   \-- ✅ esm/
|   |   |   |       |-- ✅ all-signals.d.ts
|   |   |   |       |-- ✅ all-signals.d.ts.map
|   |   |   |       |-- ✅ all-signals.js
|   |   |   |       |-- ✅ all-signals.js.map
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.d.ts.map
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ index.js.map
|   |   |   |       |-- ✅ package.json
|   |   |   |       |-- ✅ proxy-signals.d.ts
|   |   |   |       |-- ✅ proxy-signals.d.ts.map
|   |   |   |       |-- ✅ proxy-signals.js
|   |   |   |       |-- ✅ proxy-signals.js.map
|   |   |   |       |-- ✅ watchdog.d.ts
|   |   |   |       |-- ✅ watchdog.d.ts.map
|   |   |   |       |-- ✅ watchdog.js
|   |   |   |       \-- ✅ watchdog.js.map
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ form-data/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ browser.js
|   |   |   |   |-- ✅ form_data.js
|   |   |   |   \-- ✅ populate.js
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ License
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ fraction.js/
|   |   |   |-- ✅ bigfraction.js
|   |   |   |-- ✅ fraction.cjs
|   |   |   |-- ✅ fraction.d.ts
|   |   |   |-- ✅ fraction.js
|   |   |   |-- ✅ fraction.min.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ function-bind/
|   |   |   |-- ✅ .github/
|   |   |   |   |-- ✅ FUNDING.yml
|   |   |   |   \-- ✅ SECURITY.md
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ .eslintrc
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ implementation.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ functions-have-names/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ gensync/
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ .babelrc
|   |   |   |   \-- ✅ index.test.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ index.js.flow
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ get-intrinsic/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ GetIntrinsic.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ get-proto/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ Object.getPrototypeOf.d.ts
|   |   |   |-- ✅ Object.getPrototypeOf.js
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ Reflect.getPrototypeOf.d.ts
|   |   |   |-- ✅ Reflect.getPrototypeOf.js
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ glob/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ commonjs/
|   |   |   |   |   |-- ✅ glob.d.ts
|   |   |   |   |   |-- ✅ glob.d.ts.map
|   |   |   |   |   |-- ✅ glob.js
|   |   |   |   |   |-- ✅ glob.js.map
|   |   |   |   |   |-- ✅ has-magic.d.ts
|   |   |   |   |   |-- ✅ has-magic.d.ts.map
|   |   |   |   |   |-- ✅ has-magic.js
|   |   |   |   |   |-- ✅ has-magic.js.map
|   |   |   |   |   |-- ✅ ignore.d.ts
|   |   |   |   |   |-- ✅ ignore.d.ts.map
|   |   |   |   |   |-- ✅ ignore.js
|   |   |   |   |   |-- ✅ ignore.js.map
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |-- ✅ pattern.d.ts
|   |   |   |   |   |-- ✅ pattern.d.ts.map
|   |   |   |   |   |-- ✅ pattern.js
|   |   |   |   |   |-- ✅ pattern.js.map
|   |   |   |   |   |-- ✅ processor.d.ts
|   |   |   |   |   |-- ✅ processor.d.ts.map
|   |   |   |   |   |-- ✅ processor.js
|   |   |   |   |   |-- ✅ processor.js.map
|   |   |   |   |   |-- ✅ walker.d.ts
|   |   |   |   |   |-- ✅ walker.d.ts.map
|   |   |   |   |   |-- ✅ walker.js
|   |   |   |   |   \-- ✅ walker.js.map
|   |   |   |   \-- ✅ esm/
|   |   |   |       |-- ✅ bin.d.mts
|   |   |   |       |-- ✅ bin.d.mts.map
|   |   |   |       |-- ✅ bin.mjs
|   |   |   |       |-- ✅ bin.mjs.map
|   |   |   |       |-- ✅ glob.d.ts
|   |   |   |       |-- ✅ glob.d.ts.map
|   |   |   |       |-- ✅ glob.js
|   |   |   |       |-- ✅ glob.js.map
|   |   |   |       |-- ✅ has-magic.d.ts
|   |   |   |       |-- ✅ has-magic.d.ts.map
|   |   |   |       |-- ✅ has-magic.js
|   |   |   |       |-- ✅ has-magic.js.map
|   |   |   |       |-- ✅ ignore.d.ts
|   |   |   |       |-- ✅ ignore.d.ts.map
|   |   |   |       |-- ✅ ignore.js
|   |   |   |       |-- ✅ ignore.js.map
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.d.ts.map
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ index.js.map
|   |   |   |       |-- ✅ package.json
|   |   |   |       |-- ✅ pattern.d.ts
|   |   |   |       |-- ✅ pattern.d.ts.map
|   |   |   |       |-- ✅ pattern.js
|   |   |   |       |-- ✅ pattern.js.map
|   |   |   |       |-- ✅ processor.d.ts
|   |   |   |       |-- ✅ processor.d.ts.map
|   |   |   |       |-- ✅ processor.js
|   |   |   |       |-- ✅ processor.js.map
|   |   |   |       |-- ✅ walker.d.ts
|   |   |   |       |-- ✅ walker.d.ts.map
|   |   |   |       |-- ✅ walker.js
|   |   |   |       \-- ✅ walker.js.map
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ glob-parent/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ gopd/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ gOPD.d.ts
|   |   |   |-- ✅ gOPD.js
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ graceful-fs/
|   |   |   |-- ✅ clone.js
|   |   |   |-- ✅ graceful-fs.js
|   |   |   |-- ✅ legacy-streams.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ polyfills.js
|   |   |   \-- ✅ README.md
|   |   |-- ✅ has-bigints/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ has-flag/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ has-property-descriptors/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ has-symbols/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ shams/
|   |   |   |   |   |-- ✅ core-js.js
|   |   |   |   |   \-- ✅ get-own-property-symbols.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ tests.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ shams.d.ts
|   |   |   |-- ✅ shams.js
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ has-tostringtag/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ shams/
|   |   |   |   |   |-- ✅ core-js.js
|   |   |   |   |   \-- ✅ get-own-property-symbols.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ tests.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ shams.d.ts
|   |   |   |-- ✅ shams.js
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ hasown/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ html-encoding-sniffer/
|   |   |   |-- ✅ lib/
|   |   |   |   \-- ✅ html-encoding-sniffer.js
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ http-proxy-agent/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ agent.d.ts
|   |   |   |   |-- ✅ agent.js
|   |   |   |   |-- ✅ agent.js.map
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ index.js.map
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ https-proxy-agent/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ agent.d.ts
|   |   |   |   |-- ✅ agent.js
|   |   |   |   |-- ✅ agent.js.map
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ index.js.map
|   |   |   |   |-- ✅ parse-proxy-response.d.ts
|   |   |   |   |-- ✅ parse-proxy-response.js
|   |   |   |   \-- ✅ parse-proxy-response.js.map
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ iconv-lite/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ dependabot.yml
|   |   |   |-- ✅ .idea/
|   |   |   |   |-- ✅ codeStyles/
|   |   |   |   |   |-- ✅ codeStyleConfig.xml
|   |   |   |   |   \-- ✅ Project.xml
|   |   |   |   |-- ✅ inspectionProfiles/
|   |   |   |   |   \-- ✅ Project_Default.xml
|   |   |   |   |-- ✅ iconv-lite.iml
|   |   |   |   |-- ✅ modules.xml
|   |   |   |   \-- ✅ vcs.xml
|   |   |   |-- ✅ encodings/
|   |   |   |   |-- ✅ tables/
|   |   |   |   |   |-- ✅ big5-added.json
|   |   |   |   |   |-- ✅ cp936.json
|   |   |   |   |   |-- ✅ cp949.json
|   |   |   |   |   |-- ✅ cp950.json
|   |   |   |   |   |-- ✅ eucjp.json
|   |   |   |   |   |-- ✅ gb18030-ranges.json
|   |   |   |   |   |-- ✅ gbk-added.json
|   |   |   |   |   \-- ✅ shiftjis.json
|   |   |   |   |-- ✅ dbcs-codec.js
|   |   |   |   |-- ✅ dbcs-data.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ internal.js
|   |   |   |   |-- ✅ sbcs-codec.js
|   |   |   |   |-- ✅ sbcs-data-generated.js
|   |   |   |   |-- ✅ sbcs-data.js
|   |   |   |   |-- ✅ utf16.js
|   |   |   |   |-- ✅ utf32.js
|   |   |   |   \-- ✅ utf7.js
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ bom-handling.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ streams.js
|   |   |   |-- ✅ Changelog.md
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ indent-string/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ internal-slot/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .attw.json
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-arguments/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-array-buffer/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-bigint/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-binary-path/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ is-boolean-object/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-callable/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ is-core-module/
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ core.json
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ is-date-object/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-extglob/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ is-fullwidth-code-point/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ is-glob/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ is-map/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .gitattributes
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-number/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ is-number-object/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-potential-custom-element-name/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE-MIT.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ is-regex/
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-set/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .gitattributes
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-shared-array-buffer/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-string/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-symbol/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-weakmap/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ is-weakset/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .gitattributes
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ isarray/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ isexe/
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ basic.js
|   |   |   |-- ✅ .npmignore
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ mode.js
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ windows.js
|   |   |-- ✅ jackspeak/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ commonjs/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |-- ✅ parse-args-cjs.cjs.map
|   |   |   |   |   |-- ✅ parse-args-cjs.d.cts.map
|   |   |   |   |   |-- ✅ parse-args.d.ts
|   |   |   |   |   \-- ✅ parse-args.js
|   |   |   |   \-- ✅ esm/
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.d.ts.map
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ index.js.map
|   |   |   |       |-- ✅ package.json
|   |   |   |       |-- ✅ parse-args.d.ts
|   |   |   |       |-- ✅ parse-args.d.ts.map
|   |   |   |       |-- ✅ parse-args.js
|   |   |   |       \-- ✅ parse-args.js.map
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ jest-diff/
|   |   |   |-- ✅ build/
|   |   |   |   |-- ✅ cleanupSemantic.js
|   |   |   |   |-- ✅ constants.js
|   |   |   |   |-- ✅ diffLines.js
|   |   |   |   |-- ✅ diffStrings.js
|   |   |   |   |-- ✅ getAlignedDiffs.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ joinAlignedDiffs.js
|   |   |   |   |-- ✅ normalizeDiffOptions.js
|   |   |   |   |-- ✅ printDiffs.js
|   |   |   |   \-- ✅ types.js
|   |   |   |-- ✅ node_modules/
|   |   |   |   |-- ✅ ansi-styles/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ license
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   \-- ✅ readme.md
|   |   |   |   |-- ✅ pretty-format/
|   |   |   |   |   |-- ✅ build/
|   |   |   |   |   |   |-- ✅ plugins/
|   |   |   |   |   |   |   |-- ✅ lib/
|   |   |   |   |   |   |   |   |-- ✅ escapeHTML.js
|   |   |   |   |   |   |   |   \-- ✅ markup.js
|   |   |   |   |   |   |   |-- ✅ AsymmetricMatcher.js
|   |   |   |   |   |   |   |-- ✅ DOMCollection.js
|   |   |   |   |   |   |   |-- ✅ DOMElement.js
|   |   |   |   |   |   |   |-- ✅ Immutable.js
|   |   |   |   |   |   |   |-- ✅ ReactElement.js
|   |   |   |   |   |   |   \-- ✅ ReactTestComponent.js
|   |   |   |   |   |   |-- ✅ collections.js
|   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ types.js
|   |   |   |   |   |-- ✅ LICENSE
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   \-- ✅ README.md
|   |   |   |   \-- ✅ react-is/
|   |   |   |       |-- ✅ cjs/
|   |   |   |       |   |-- ✅ react-is.development.js
|   |   |   |       |   \-- ✅ react-is.production.min.js
|   |   |   |       |-- ✅ umd/
|   |   |   |       |   |-- ✅ react-is.development.js
|   |   |   |       |   \-- ✅ react-is.production.min.js
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ LICENSE
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ README.md
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ jest-get-type/
|   |   |   |-- ✅ build/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   \-- ✅ package.json
|   |   |-- ✅ jest-matcher-utils/
|   |   |   |-- ✅ build/
|   |   |   |   |-- ✅ deepCyclicCopyReplaceable.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ Replaceable.js
|   |   |   |-- ✅ node_modules/
|   |   |   |   |-- ✅ ansi-styles/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ license
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   \-- ✅ readme.md
|   |   |   |   |-- ✅ pretty-format/
|   |   |   |   |   |-- ✅ build/
|   |   |   |   |   |   |-- ✅ plugins/
|   |   |   |   |   |   |   |-- ✅ lib/
|   |   |   |   |   |   |   |   |-- ✅ escapeHTML.js
|   |   |   |   |   |   |   |   \-- ✅ markup.js
|   |   |   |   |   |   |   |-- ✅ AsymmetricMatcher.js
|   |   |   |   |   |   |   |-- ✅ DOMCollection.js
|   |   |   |   |   |   |   |-- ✅ DOMElement.js
|   |   |   |   |   |   |   |-- ✅ Immutable.js
|   |   |   |   |   |   |   |-- ✅ ReactElement.js
|   |   |   |   |   |   |   \-- ✅ ReactTestComponent.js
|   |   |   |   |   |   |-- ✅ collections.js
|   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ types.js
|   |   |   |   |   |-- ✅ LICENSE
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   \-- ✅ README.md
|   |   |   |   \-- ✅ react-is/
|   |   |   |       |-- ✅ cjs/
|   |   |   |       |   |-- ✅ react-is.development.js
|   |   |   |       |   \-- ✅ react-is.production.min.js
|   |   |   |       |-- ✅ umd/
|   |   |   |       |   |-- ✅ react-is.development.js
|   |   |   |       |   \-- ✅ react-is.production.min.js
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ LICENSE
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ README.md
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ jest-message-util/
|   |   |   |-- ✅ build/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ types.js
|   |   |   |-- ✅ node_modules/
|   |   |   |   |-- ✅ ansi-styles/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ license
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   \-- ✅ readme.md
|   |   |   |   |-- ✅ pretty-format/
|   |   |   |   |   |-- ✅ build/
|   |   |   |   |   |   |-- ✅ plugins/
|   |   |   |   |   |   |   |-- ✅ lib/
|   |   |   |   |   |   |   |   |-- ✅ escapeHTML.js
|   |   |   |   |   |   |   |   \-- ✅ markup.js
|   |   |   |   |   |   |   |-- ✅ AsymmetricMatcher.js
|   |   |   |   |   |   |   |-- ✅ DOMCollection.js
|   |   |   |   |   |   |   |-- ✅ DOMElement.js
|   |   |   |   |   |   |   |-- ✅ Immutable.js
|   |   |   |   |   |   |   |-- ✅ ReactElement.js
|   |   |   |   |   |   |   \-- ✅ ReactTestComponent.js
|   |   |   |   |   |   |-- ✅ collections.js
|   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ types.js
|   |   |   |   |   |-- ✅ LICENSE
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   \-- ✅ README.md
|   |   |   |   \-- ✅ react-is/
|   |   |   |       |-- ✅ cjs/
|   |   |   |       |   |-- ✅ react-is.development.js
|   |   |   |       |   \-- ✅ react-is.production.min.js
|   |   |   |       |-- ✅ umd/
|   |   |   |       |   |-- ✅ react-is.development.js
|   |   |   |       |   \-- ✅ react-is.production.min.js
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ LICENSE
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ README.md
|   |   |   |-- ✅ LICENSE
|   |   |   \-- ✅ package.json
|   |   |-- ✅ jest-util/
|   |   |   |-- ✅ build/
|   |   |   |   |-- ✅ clearLine.js
|   |   |   |   |-- ✅ convertDescriptorToString.js
|   |   |   |   |-- ✅ createDirectory.js
|   |   |   |   |-- ✅ createProcessObject.js
|   |   |   |   |-- ✅ deepCyclicCopy.js
|   |   |   |   |-- ✅ ErrorWithStack.js
|   |   |   |   |-- ✅ formatTime.js
|   |   |   |   |-- ✅ globsToMatcher.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ installCommonGlobals.js
|   |   |   |   |-- ✅ interopRequireDefault.js
|   |   |   |   |-- ✅ invariant.js
|   |   |   |   |-- ✅ isInteractive.js
|   |   |   |   |-- ✅ isNonNullable.js
|   |   |   |   |-- ✅ isPromise.js
|   |   |   |   |-- ✅ pluralize.js
|   |   |   |   |-- ✅ preRunMessage.js
|   |   |   |   |-- ✅ replacePathSepForGlob.js
|   |   |   |   |-- ✅ requireOrImportModule.js
|   |   |   |   |-- ✅ setGlobal.js
|   |   |   |   |-- ✅ specialChars.js
|   |   |   |   |-- ✅ testPathPatternToRegExp.js
|   |   |   |   \-- ✅ tryRealpath.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ Readme.md
|   |   |-- ✅ jiti/
|   |   |   |-- ✅ bin/
|   |   |   |   \-- ✅ jiti.js
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ plugins/
|   |   |   |   |   |-- ✅ babel-plugin-transform-import-meta.d.ts
|   |   |   |   |   \-- ✅ import-meta-env.d.ts
|   |   |   |   |-- ✅ babel.d.ts
|   |   |   |   |-- ✅ babel.js
|   |   |   |   |-- ✅ jiti.d.ts
|   |   |   |   |-- ✅ jiti.js
|   |   |   |   |-- ✅ types.d.ts
|   |   |   |   \-- ✅ utils.d.ts
|   |   |   |-- ✅ lib/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ register.js
|   |   |-- ✅ js-tokens/
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ jsdom/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ jsdom/
|   |   |   |   |   |-- ✅ browser/
|   |   |   |   |   |   |-- ✅ parser/
|   |   |   |   |   |   |   |-- ✅ html.js
|   |   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |   \-- ✅ xml.js
|   |   |   |   |   |   |-- ✅ resources/
|   |   |   |   |   |   |   |-- ✅ async-resource-queue.js
|   |   |   |   |   |   |   |-- ✅ no-op-resource-loader.js
|   |   |   |   |   |   |   |-- ✅ per-document-resource-loader.js
|   |   |   |   |   |   |   |-- ✅ request-manager.js
|   |   |   |   |   |   |   |-- ✅ resource-loader.js
|   |   |   |   |   |   |   \-- ✅ resource-queue.js
|   |   |   |   |   |   |-- ✅ default-stylesheet.js
|   |   |   |   |   |   |-- ✅ js-globals.json
|   |   |   |   |   |   |-- ✅ not-implemented.js
|   |   |   |   |   |   \-- ✅ Window.js
|   |   |   |   |   |-- ✅ level2/
|   |   |   |   |   |   \-- ✅ style.js
|   |   |   |   |   |-- ✅ level3/
|   |   |   |   |   |   \-- ✅ xpath.js
|   |   |   |   |   |-- ✅ living/
|   |   |   |   |   |   |-- ✅ aborting/
|   |   |   |   |   |   |   |-- ✅ AbortController-impl.js
|   |   |   |   |   |   |   \-- ✅ AbortSignal-impl.js
|   |   |   |   |   |   |-- ✅ attributes/
|   |   |   |   |   |   |   |-- ✅ Attr-impl.js
|   |   |   |   |   |   |   \-- ✅ NamedNodeMap-impl.js
|   |   |   |   |   |   |-- ✅ constraint-validation/
|   |   |   |   |   |   |   |-- ✅ DefaultConstraintValidation-impl.js
|   |   |   |   |   |   |   \-- ✅ ValidityState-impl.js
|   |   |   |   |   |   |-- ✅ crypto/
|   |   |   |   |   |   |   \-- ✅ Crypto-impl.js
|   |   |   |   |   |   |-- ✅ cssom/
|   |   |   |   |   |   |   \-- ✅ StyleSheetList-impl.js
|   |   |   |   |   |   |-- ✅ custom-elements/
|   |   |   |   |   |   |   \-- ✅ CustomElementRegistry-impl.js
|   |   |   |   |   |   |-- ✅ domparsing/
|   |   |   |   |   |   |   |-- ✅ DOMParser-impl.js
|   |   |   |   |   |   |   |-- ✅ InnerHTML-impl.js
|   |   |   |   |   |   |   |-- ✅ parse5-adapter-serialization.js
|   |   |   |   |   |   |   |-- ✅ serialization.js
|   |   |   |   |   |   |   \-- ✅ XMLSerializer-impl.js
|   |   |   |   |   |   |-- ✅ events/
|   |   |   |   |   |   |   |-- ✅ CloseEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ CompositionEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ CustomEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ ErrorEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ Event-impl.js
|   |   |   |   |   |   |   |-- ✅ EventModifierMixin-impl.js
|   |   |   |   |   |   |   |-- ✅ EventTarget-impl.js
|   |   |   |   |   |   |   |-- ✅ FocusEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ HashChangeEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ InputEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ KeyboardEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ MessageEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ MouseEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ PageTransitionEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ PopStateEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ ProgressEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ StorageEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ SubmitEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ TouchEvent-impl.js
|   |   |   |   |   |   |   |-- ✅ UIEvent-impl.js
|   |   |   |   |   |   |   \-- ✅ WheelEvent-impl.js
|   |   |   |   |   |   |-- ✅ fetch/
|   |   |   |   |   |   |   |-- ✅ header-list.js
|   |   |   |   |   |   |   |-- ✅ header-types.js
|   |   |   |   |   |   |   \-- ✅ Headers-impl.js
|   |   |   |   |   |   |-- ✅ file-api/
|   |   |   |   |   |   |   |-- ✅ Blob-impl.js
|   |   |   |   |   |   |   |-- ✅ File-impl.js
|   |   |   |   |   |   |   |-- ✅ FileList-impl.js
|   |   |   |   |   |   |   \-- ✅ FileReader-impl.js
|   |   |   |   |   |   |-- ✅ generated/
|   |   |   |   |   |   |   |-- ✅ AbortController.js
|   |   |   |   |   |   |   |-- ✅ AbortSignal.js
|   |   |   |   |   |   |   |-- ✅ AbstractRange.js
|   |   |   |   |   |   |   |-- ✅ AddEventListenerOptions.js
|   |   |   |   |   |   |   |-- ✅ AssignedNodesOptions.js
|   |   |   |   |   |   |   |-- ✅ Attr.js
|   |   |   |   |   |   |   |-- ✅ BarProp.js
|   |   |   |   |   |   |   |-- ✅ BinaryType.js
|   |   |   |   |   |   |   |-- ✅ Blob.js
|   |   |   |   |   |   |   |-- ✅ BlobCallback.js
|   |   |   |   |   |   |   |-- ✅ BlobPropertyBag.js
|   |   |   |   |   |   |   |-- ✅ CanPlayTypeResult.js
|   |   |   |   |   |   |   |-- ✅ CDATASection.js
|   |   |   |   |   |   |   |-- ✅ CharacterData.js
|   |   |   |   |   |   |   |-- ✅ CloseEvent.js
|   |   |   |   |   |   |   |-- ✅ CloseEventInit.js
|   |   |   |   |   |   |   |-- ✅ Comment.js
|   |   |   |   |   |   |   |-- ✅ CompositionEvent.js
|   |   |   |   |   |   |   |-- ✅ CompositionEventInit.js
|   |   |   |   |   |   |   |-- ✅ Crypto.js
|   |   |   |   |   |   |   |-- ✅ CustomElementConstructor.js
|   |   |   |   |   |   |   |-- ✅ CustomElementRegistry.js
|   |   |   |   |   |   |   |-- ✅ CustomEvent.js
|   |   |   |   |   |   |   |-- ✅ CustomEventInit.js
|   |   |   |   |   |   |   |-- ✅ Document.js
|   |   |   |   |   |   |   |-- ✅ DocumentFragment.js
|   |   |   |   |   |   |   |-- ✅ DocumentReadyState.js
|   |   |   |   |   |   |   |-- ✅ DocumentType.js
|   |   |   |   |   |   |   |-- ✅ DOMImplementation.js
|   |   |   |   |   |   |   |-- ✅ DOMParser.js
|   |   |   |   |   |   |   |-- ✅ DOMRect.js
|   |   |   |   |   |   |   |-- ✅ DOMRectInit.js
|   |   |   |   |   |   |   |-- ✅ DOMRectReadOnly.js
|   |   |   |   |   |   |   |-- ✅ DOMStringMap.js
|   |   |   |   |   |   |   |-- ✅ DOMTokenList.js
|   |   |   |   |   |   |   |-- ✅ Element.js
|   |   |   |   |   |   |   |-- ✅ ElementCreationOptions.js
|   |   |   |   |   |   |   |-- ✅ ElementDefinitionOptions.js
|   |   |   |   |   |   |   |-- ✅ EndingType.js
|   |   |   |   |   |   |   |-- ✅ ErrorEvent.js
|   |   |   |   |   |   |   |-- ✅ ErrorEventInit.js
|   |   |   |   |   |   |   |-- ✅ Event.js
|   |   |   |   |   |   |   |-- ✅ EventHandlerNonNull.js
|   |   |   |   |   |   |   |-- ✅ EventInit.js
|   |   |   |   |   |   |   |-- ✅ EventListener.js
|   |   |   |   |   |   |   |-- ✅ EventListenerOptions.js
|   |   |   |   |   |   |   |-- ✅ EventModifierInit.js
|   |   |   |   |   |   |   |-- ✅ EventTarget.js
|   |   |   |   |   |   |   |-- ✅ External.js
|   |   |   |   |   |   |   |-- ✅ File.js
|   |   |   |   |   |   |   |-- ✅ FileList.js
|   |   |   |   |   |   |   |-- ✅ FilePropertyBag.js
|   |   |   |   |   |   |   |-- ✅ FileReader.js
|   |   |   |   |   |   |   |-- ✅ FocusEvent.js
|   |   |   |   |   |   |   |-- ✅ FocusEventInit.js
|   |   |   |   |   |   |   |-- ✅ FormData.js
|   |   |   |   |   |   |   |-- ✅ Function.js
|   |   |   |   |   |   |   |-- ✅ GetRootNodeOptions.js
|   |   |   |   |   |   |   |-- ✅ HashChangeEvent.js
|   |   |   |   |   |   |   |-- ✅ HashChangeEventInit.js
|   |   |   |   |   |   |   |-- ✅ Headers.js
|   |   |   |   |   |   |   |-- ✅ History.js
|   |   |   |   |   |   |   |-- ✅ HTMLAnchorElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLAreaElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLAudioElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLBaseElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLBodyElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLBRElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLButtonElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLCanvasElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLCollection.js
|   |   |   |   |   |   |   |-- ✅ HTMLDataElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLDataListElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLDetailsElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLDialogElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLDirectoryElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLDivElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLDListElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLEmbedElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLFieldSetElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLFontElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLFormControlsCollection.js
|   |   |   |   |   |   |   |-- ✅ HTMLFormElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLFrameElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLFrameSetElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLHeadElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLHeadingElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLHRElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLHtmlElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLIFrameElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLImageElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLInputElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLLabelElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLLegendElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLLIElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLLinkElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLMapElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLMarqueeElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLMediaElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLMenuElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLMetaElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLMeterElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLModElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLObjectElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLOListElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLOptGroupElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLOptionElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLOptionsCollection.js
|   |   |   |   |   |   |   |-- ✅ HTMLOutputElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLParagraphElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLParamElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLPictureElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLPreElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLProgressElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLQuoteElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLScriptElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLSelectElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLSlotElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLSourceElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLSpanElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLStyleElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableCaptionElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableCellElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableColElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableRowElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableSectionElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLTemplateElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLTextAreaElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLTimeElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLTitleElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLTrackElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLUListElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLUnknownElement.js
|   |   |   |   |   |   |   |-- ✅ HTMLVideoElement.js
|   |   |   |   |   |   |   |-- ✅ InputEvent.js
|   |   |   |   |   |   |   |-- ✅ InputEventInit.js
|   |   |   |   |   |   |   |-- ✅ KeyboardEvent.js
|   |   |   |   |   |   |   |-- ✅ KeyboardEventInit.js
|   |   |   |   |   |   |   |-- ✅ Location.js
|   |   |   |   |   |   |   |-- ✅ MessageEvent.js
|   |   |   |   |   |   |   |-- ✅ MessageEventInit.js
|   |   |   |   |   |   |   |-- ✅ MimeType.js
|   |   |   |   |   |   |   |-- ✅ MimeTypeArray.js
|   |   |   |   |   |   |   |-- ✅ MouseEvent.js
|   |   |   |   |   |   |   |-- ✅ MouseEventInit.js
|   |   |   |   |   |   |   |-- ✅ MutationCallback.js
|   |   |   |   |   |   |   |-- ✅ MutationObserver.js
|   |   |   |   |   |   |   |-- ✅ MutationObserverInit.js
|   |   |   |   |   |   |   |-- ✅ MutationRecord.js
|   |   |   |   |   |   |   |-- ✅ NamedNodeMap.js
|   |   |   |   |   |   |   |-- ✅ Navigator.js
|   |   |   |   |   |   |   |-- ✅ Node.js
|   |   |   |   |   |   |   |-- ✅ NodeFilter.js
|   |   |   |   |   |   |   |-- ✅ NodeIterator.js
|   |   |   |   |   |   |   |-- ✅ NodeList.js
|   |   |   |   |   |   |   |-- ✅ OnBeforeUnloadEventHandlerNonNull.js
|   |   |   |   |   |   |   |-- ✅ OnErrorEventHandlerNonNull.js
|   |   |   |   |   |   |   |-- ✅ PageTransitionEvent.js
|   |   |   |   |   |   |   |-- ✅ PageTransitionEventInit.js
|   |   |   |   |   |   |   |-- ✅ Performance.js
|   |   |   |   |   |   |   |-- ✅ Plugin.js
|   |   |   |   |   |   |   |-- ✅ PluginArray.js
|   |   |   |   |   |   |   |-- ✅ PopStateEvent.js
|   |   |   |   |   |   |   |-- ✅ PopStateEventInit.js
|   |   |   |   |   |   |   |-- ✅ ProcessingInstruction.js
|   |   |   |   |   |   |   |-- ✅ ProgressEvent.js
|   |   |   |   |   |   |   |-- ✅ ProgressEventInit.js
|   |   |   |   |   |   |   |-- ✅ RadioNodeList.js
|   |   |   |   |   |   |   |-- ✅ Range.js
|   |   |   |   |   |   |   |-- ✅ Screen.js
|   |   |   |   |   |   |   |-- ✅ ScrollBehavior.js
|   |   |   |   |   |   |   |-- ✅ ScrollIntoViewOptions.js
|   |   |   |   |   |   |   |-- ✅ ScrollLogicalPosition.js
|   |   |   |   |   |   |   |-- ✅ ScrollOptions.js
|   |   |   |   |   |   |   |-- ✅ ScrollRestoration.js
|   |   |   |   |   |   |   |-- ✅ Selection.js
|   |   |   |   |   |   |   |-- ✅ SelectionMode.js
|   |   |   |   |   |   |   |-- ✅ ShadowRoot.js
|   |   |   |   |   |   |   |-- ✅ ShadowRootInit.js
|   |   |   |   |   |   |   |-- ✅ ShadowRootMode.js
|   |   |   |   |   |   |   |-- ✅ StaticRange.js
|   |   |   |   |   |   |   |-- ✅ StaticRangeInit.js
|   |   |   |   |   |   |   |-- ✅ Storage.js
|   |   |   |   |   |   |   |-- ✅ StorageEvent.js
|   |   |   |   |   |   |   |-- ✅ StorageEventInit.js
|   |   |   |   |   |   |   |-- ✅ StyleSheetList.js
|   |   |   |   |   |   |   |-- ✅ SubmitEvent.js
|   |   |   |   |   |   |   |-- ✅ SubmitEventInit.js
|   |   |   |   |   |   |   |-- ✅ SupportedType.js
|   |   |   |   |   |   |   |-- ✅ SVGAnimatedString.js
|   |   |   |   |   |   |   |-- ✅ SVGBoundingBoxOptions.js
|   |   |   |   |   |   |   |-- ✅ SVGElement.js
|   |   |   |   |   |   |   |-- ✅ SVGGraphicsElement.js
|   |   |   |   |   |   |   |-- ✅ SVGNumber.js
|   |   |   |   |   |   |   |-- ✅ SVGStringList.js
|   |   |   |   |   |   |   |-- ✅ SVGSVGElement.js
|   |   |   |   |   |   |   |-- ✅ SVGTitleElement.js
|   |   |   |   |   |   |   |-- ✅ Text.js
|   |   |   |   |   |   |   |-- ✅ TextTrackKind.js
|   |   |   |   |   |   |   |-- ✅ TouchEvent.js
|   |   |   |   |   |   |   |-- ✅ TouchEventInit.js
|   |   |   |   |   |   |   |-- ✅ TreeWalker.js
|   |   |   |   |   |   |   |-- ✅ UIEvent.js
|   |   |   |   |   |   |   |-- ✅ UIEventInit.js
|   |   |   |   |   |   |   |-- ✅ utils.js
|   |   |   |   |   |   |   |-- ✅ ValidityState.js
|   |   |   |   |   |   |   |-- ✅ VisibilityState.js
|   |   |   |   |   |   |   |-- ✅ VoidFunction.js
|   |   |   |   |   |   |   |-- ✅ WebSocket.js
|   |   |   |   |   |   |   |-- ✅ WheelEvent.js
|   |   |   |   |   |   |   |-- ✅ WheelEventInit.js
|   |   |   |   |   |   |   |-- ✅ XMLDocument.js
|   |   |   |   |   |   |   |-- ✅ XMLHttpRequest.js
|   |   |   |   |   |   |   |-- ✅ XMLHttpRequestEventTarget.js
|   |   |   |   |   |   |   |-- ✅ XMLHttpRequestResponseType.js
|   |   |   |   |   |   |   |-- ✅ XMLHttpRequestUpload.js
|   |   |   |   |   |   |   \-- ✅ XMLSerializer.js
|   |   |   |   |   |   |-- ✅ geometry/
|   |   |   |   |   |   |   |-- ✅ DOMRect-impl.js
|   |   |   |   |   |   |   \-- ✅ DOMRectReadOnly-impl.js
|   |   |   |   |   |   |-- ✅ helpers/
|   |   |   |   |   |   |   |-- ✅ svg/
|   |   |   |   |   |   |   |   |-- ✅ basic-types.js
|   |   |   |   |   |   |   |   \-- ✅ render.js
|   |   |   |   |   |   |   |-- ✅ agent-factory.js
|   |   |   |   |   |   |   |-- ✅ binary-data.js
|   |   |   |   |   |   |   |-- ✅ colors.js
|   |   |   |   |   |   |   |-- ✅ create-element.js
|   |   |   |   |   |   |   |-- ✅ create-event-accessor.js
|   |   |   |   |   |   |   |-- ✅ custom-elements.js
|   |   |   |   |   |   |   |-- ✅ dates-and-times.js
|   |   |   |   |   |   |   |-- ✅ details.js
|   |   |   |   |   |   |   |-- ✅ document-base-url.js
|   |   |   |   |   |   |   |-- ✅ events.js
|   |   |   |   |   |   |   |-- ✅ focusing.js
|   |   |   |   |   |   |   |-- ✅ form-controls.js
|   |   |   |   |   |   |   |-- ✅ html-constructor.js
|   |   |   |   |   |   |   |-- ✅ http-request.js
|   |   |   |   |   |   |   |-- ✅ internal-constants.js
|   |   |   |   |   |   |   |-- ✅ iterable-weak-set.js
|   |   |   |   |   |   |   |-- ✅ json.js
|   |   |   |   |   |   |   |-- ✅ mutation-observers.js
|   |   |   |   |   |   |   |-- ✅ namespaces.js
|   |   |   |   |   |   |   |-- ✅ node.js
|   |   |   |   |   |   |   |-- ✅ number-and-date-inputs.js
|   |   |   |   |   |   |   |-- ✅ ordered-set.js
|   |   |   |   |   |   |   |-- ✅ page-transition-event.js
|   |   |   |   |   |   |   |-- ✅ runtime-script-errors.js
|   |   |   |   |   |   |   |-- ✅ selectors.js
|   |   |   |   |   |   |   |-- ✅ shadow-dom.js
|   |   |   |   |   |   |   |-- ✅ strings.js
|   |   |   |   |   |   |   |-- ✅ style-rules.js
|   |   |   |   |   |   |   |-- ✅ stylesheets.js
|   |   |   |   |   |   |   |-- ✅ text.js
|   |   |   |   |   |   |   |-- ✅ traversal.js
|   |   |   |   |   |   |   \-- ✅ validate-names.js
|   |   |   |   |   |   |-- ✅ hr-time/
|   |   |   |   |   |   |   \-- ✅ Performance-impl.js
|   |   |   |   |   |   |-- ✅ mutation-observer/
|   |   |   |   |   |   |   |-- ✅ MutationObserver-impl.js
|   |   |   |   |   |   |   \-- ✅ MutationRecord-impl.js
|   |   |   |   |   |   |-- ✅ navigator/
|   |   |   |   |   |   |   |-- ✅ MimeType-impl.js
|   |   |   |   |   |   |   |-- ✅ MimeTypeArray-impl.js
|   |   |   |   |   |   |   |-- ✅ Navigator-impl.js
|   |   |   |   |   |   |   |-- ✅ NavigatorConcurrentHardware-impl.js
|   |   |   |   |   |   |   |-- ✅ NavigatorCookies-impl.js
|   |   |   |   |   |   |   |-- ✅ NavigatorID-impl.js
|   |   |   |   |   |   |   |-- ✅ NavigatorLanguage-impl.js
|   |   |   |   |   |   |   |-- ✅ NavigatorOnLine-impl.js
|   |   |   |   |   |   |   |-- ✅ NavigatorPlugins-impl.js
|   |   |   |   |   |   |   |-- ✅ Plugin-impl.js
|   |   |   |   |   |   |   \-- ✅ PluginArray-impl.js
|   |   |   |   |   |   |-- ✅ nodes/
|   |   |   |   |   |   |   |-- ✅ CDATASection-impl.js
|   |   |   |   |   |   |   |-- ✅ CharacterData-impl.js
|   |   |   |   |   |   |   |-- ✅ ChildNode-impl.js
|   |   |   |   |   |   |   |-- ✅ Comment-impl.js
|   |   |   |   |   |   |   |-- ✅ Document-impl.js
|   |   |   |   |   |   |   |-- ✅ DocumentFragment-impl.js
|   |   |   |   |   |   |   |-- ✅ DocumentOrShadowRoot-impl.js
|   |   |   |   |   |   |   |-- ✅ DocumentType-impl.js
|   |   |   |   |   |   |   |-- ✅ DOMImplementation-impl.js
|   |   |   |   |   |   |   |-- ✅ DOMStringMap-impl.js
|   |   |   |   |   |   |   |-- ✅ DOMTokenList-impl.js
|   |   |   |   |   |   |   |-- ✅ Element-impl.js
|   |   |   |   |   |   |   |-- ✅ ElementContentEditable-impl.js
|   |   |   |   |   |   |   |-- ✅ ElementCSSInlineStyle-impl.js
|   |   |   |   |   |   |   |-- ✅ GlobalEventHandlers-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLAnchorElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLAreaElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLAudioElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLBaseElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLBodyElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLBRElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLButtonElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLCanvasElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLCollection-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLDataElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLDataListElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLDetailsElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLDialogElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLDirectoryElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLDivElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLDListElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLEmbedElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLFieldSetElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLFontElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLFormControlsCollection-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLFormElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLFrameElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLFrameSetElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLHeadElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLHeadingElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLHRElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLHtmlElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLHyperlinkElementUtils-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLIFrameElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLImageElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLInputElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLLabelElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLLegendElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLLIElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLLinkElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLMapElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLMarqueeElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLMediaElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLMenuElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLMetaElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLMeterElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLModElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLObjectElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLOListElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLOptGroupElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLOptionElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLOptionsCollection-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLOrSVGElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLOutputElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLParagraphElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLParamElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLPictureElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLPreElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLProgressElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLQuoteElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLScriptElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLSelectElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLSlotElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLSourceElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLSpanElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLStyleElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableCaptionElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableCellElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableColElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableRowElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLTableSectionElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLTemplateElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLTextAreaElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLTimeElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLTitleElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLTrackElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLUListElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLUnknownElement-impl.js
|   |   |   |   |   |   |   |-- ✅ HTMLVideoElement-impl.js
|   |   |   |   |   |   |   |-- ✅ LinkStyle-impl.js
|   |   |   |   |   |   |   |-- ✅ Node-impl.js
|   |   |   |   |   |   |   |-- ✅ NodeList-impl.js
|   |   |   |   |   |   |   |-- ✅ NonDocumentTypeChildNode-impl.js
|   |   |   |   |   |   |   |-- ✅ NonElementParentNode-impl.js
|   |   |   |   |   |   |   |-- ✅ ParentNode-impl.js
|   |   |   |   |   |   |   |-- ✅ ProcessingInstruction-impl.js
|   |   |   |   |   |   |   |-- ✅ RadioNodeList-impl.js
|   |   |   |   |   |   |   |-- ✅ ShadowRoot-impl.js
|   |   |   |   |   |   |   |-- ✅ Slotable-impl.js
|   |   |   |   |   |   |   |-- ✅ SVGElement-impl.js
|   |   |   |   |   |   |   |-- ✅ SVGGraphicsElement-impl.js
|   |   |   |   |   |   |   |-- ✅ SVGSVGElement-impl.js
|   |   |   |   |   |   |   |-- ✅ SVGTests-impl.js
|   |   |   |   |   |   |   |-- ✅ SVGTitleElement-impl.js
|   |   |   |   |   |   |   |-- ✅ Text-impl.js
|   |   |   |   |   |   |   |-- ✅ WindowEventHandlers-impl.js
|   |   |   |   |   |   |   \-- ✅ XMLDocument-impl.js
|   |   |   |   |   |   |-- ✅ range/
|   |   |   |   |   |   |   |-- ✅ AbstractRange-impl.js
|   |   |   |   |   |   |   |-- ✅ boundary-point.js
|   |   |   |   |   |   |   |-- ✅ Range-impl.js
|   |   |   |   |   |   |   \-- ✅ StaticRange-impl.js
|   |   |   |   |   |   |-- ✅ selection/
|   |   |   |   |   |   |   \-- ✅ Selection-impl.js
|   |   |   |   |   |   |-- ✅ svg/
|   |   |   |   |   |   |   |-- ✅ SVGAnimatedString-impl.js
|   |   |   |   |   |   |   |-- ✅ SVGListBase.js
|   |   |   |   |   |   |   |-- ✅ SVGNumber-impl.js
|   |   |   |   |   |   |   \-- ✅ SVGStringList-impl.js
|   |   |   |   |   |   |-- ✅ traversal/
|   |   |   |   |   |   |   |-- ✅ helpers.js
|   |   |   |   |   |   |   |-- ✅ NodeIterator-impl.js
|   |   |   |   |   |   |   \-- ✅ TreeWalker-impl.js
|   |   |   |   |   |   |-- ✅ websockets/
|   |   |   |   |   |   |   \-- ✅ WebSocket-impl.js
|   |   |   |   |   |   |-- ✅ webstorage/
|   |   |   |   |   |   |   \-- ✅ Storage-impl.js
|   |   |   |   |   |   |-- ✅ window/
|   |   |   |   |   |   |   |-- ✅ BarProp-impl.js
|   |   |   |   |   |   |   |-- ✅ External-impl.js
|   |   |   |   |   |   |   |-- ✅ History-impl.js
|   |   |   |   |   |   |   |-- ✅ Location-impl.js
|   |   |   |   |   |   |   |-- ✅ navigation.js
|   |   |   |   |   |   |   |-- ✅ Screen-impl.js
|   |   |   |   |   |   |   \-- ✅ SessionHistory.js
|   |   |   |   |   |   |-- ✅ xhr/
|   |   |   |   |   |   |   |-- ✅ FormData-impl.js
|   |   |   |   |   |   |   |-- ✅ xhr-sync-worker.js
|   |   |   |   |   |   |   |-- ✅ xhr-utils.js
|   |   |   |   |   |   |   |-- ✅ XMLHttpRequest-impl.js
|   |   |   |   |   |   |   |-- ✅ XMLHttpRequestEventTarget-impl.js
|   |   |   |   |   |   |   \-- ✅ XMLHttpRequestUpload-impl.js
|   |   |   |   |   |   |-- ✅ attributes.js
|   |   |   |   |   |   |-- ✅ documents.js
|   |   |   |   |   |   |-- ✅ interfaces.js
|   |   |   |   |   |   |-- ✅ named-properties-window.js
|   |   |   |   |   |   |-- ✅ node-document-position.js
|   |   |   |   |   |   |-- ✅ node-type.js
|   |   |   |   |   |   |-- ✅ node.js
|   |   |   |   |   |   \-- ✅ post-message.js
|   |   |   |   |   |-- ✅ named-properties-tracker.js
|   |   |   |   |   |-- ✅ utils.js
|   |   |   |   |   \-- ✅ virtual-console.js
|   |   |   |   \-- ✅ api.js
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ jsesc/
|   |   |   |-- ✅ bin/
|   |   |   |   \-- ✅ jsesc
|   |   |   |-- ✅ man/
|   |   |   |   \-- ✅ jsesc.1
|   |   |   |-- ✅ jsesc.js
|   |   |   |-- ✅ LICENSE-MIT.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ json5/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ index.min.js
|   |   |   |   |-- ✅ index.min.mjs
|   |   |   |   \-- ✅ index.mjs
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ cli.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ parse.d.ts
|   |   |   |   |-- ✅ parse.js
|   |   |   |   |-- ✅ register.js
|   |   |   |   |-- ✅ require.js
|   |   |   |   |-- ✅ stringify.d.ts
|   |   |   |   |-- ✅ stringify.js
|   |   |   |   |-- ✅ unicode.d.ts
|   |   |   |   |-- ✅ unicode.js
|   |   |   |   |-- ✅ util.d.ts
|   |   |   |   \-- ✅ util.js
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ lilconfig/
|   |   |   |-- ✅ src/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ lines-and-columns/
|   |   |   |-- ✅ build/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ loose-envify/
|   |   |   |-- ✅ cli.js
|   |   |   |-- ✅ custom.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ loose-envify.js
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ replace.js
|   |   |-- ✅ loupe/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ arguments.d.ts
|   |   |   |   |-- ✅ arguments.d.ts.map
|   |   |   |   |-- ✅ arguments.js
|   |   |   |   |-- ✅ array.d.ts
|   |   |   |   |-- ✅ array.d.ts.map
|   |   |   |   |-- ✅ array.js
|   |   |   |   |-- ✅ bigint.d.ts
|   |   |   |   |-- ✅ bigint.d.ts.map
|   |   |   |   |-- ✅ bigint.js
|   |   |   |   |-- ✅ class.d.ts
|   |   |   |   |-- ✅ class.d.ts.map
|   |   |   |   |-- ✅ class.js
|   |   |   |   |-- ✅ date.d.ts
|   |   |   |   |-- ✅ date.d.ts.map
|   |   |   |   |-- ✅ date.js
|   |   |   |   |-- ✅ error.d.ts
|   |   |   |   |-- ✅ error.d.ts.map
|   |   |   |   |-- ✅ error.js
|   |   |   |   |-- ✅ function.d.ts
|   |   |   |   |-- ✅ function.d.ts.map
|   |   |   |   |-- ✅ function.js
|   |   |   |   |-- ✅ helpers.d.ts
|   |   |   |   |-- ✅ helpers.d.ts.map
|   |   |   |   |-- ✅ helpers.js
|   |   |   |   |-- ✅ html.d.ts
|   |   |   |   |-- ✅ html.d.ts.map
|   |   |   |   |-- ✅ html.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ map.d.ts
|   |   |   |   |-- ✅ map.d.ts.map
|   |   |   |   |-- ✅ map.js
|   |   |   |   |-- ✅ number.d.ts
|   |   |   |   |-- ✅ number.d.ts.map
|   |   |   |   |-- ✅ number.js
|   |   |   |   |-- ✅ object.d.ts
|   |   |   |   |-- ✅ object.d.ts.map
|   |   |   |   |-- ✅ object.js
|   |   |   |   |-- ✅ promise.d.ts
|   |   |   |   |-- ✅ promise.d.ts.map
|   |   |   |   |-- ✅ promise.js
|   |   |   |   |-- ✅ regexp.d.ts
|   |   |   |   |-- ✅ regexp.d.ts.map
|   |   |   |   |-- ✅ regexp.js
|   |   |   |   |-- ✅ set.d.ts
|   |   |   |   |-- ✅ set.d.ts.map
|   |   |   |   |-- ✅ set.js
|   |   |   |   |-- ✅ string.d.ts
|   |   |   |   |-- ✅ string.d.ts.map
|   |   |   |   |-- ✅ string.js
|   |   |   |   |-- ✅ symbol.d.ts
|   |   |   |   |-- ✅ symbol.d.ts.map
|   |   |   |   |-- ✅ symbol.js
|   |   |   |   |-- ✅ typedarray.d.ts
|   |   |   |   |-- ✅ typedarray.d.ts.map
|   |   |   |   |-- ✅ typedarray.js
|   |   |   |   |-- ✅ types.d.ts
|   |   |   |   |-- ✅ types.d.ts.map
|   |   |   |   \-- ✅ types.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ loupe.js
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ lru-cache/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ lz-string/
|   |   |   |-- ✅ bin/
|   |   |   |   \-- ✅ bin.js
|   |   |   |-- ✅ libs/
|   |   |   |   |-- ✅ base64-string.js
|   |   |   |   |-- ✅ lz-string.js
|   |   |   |   \-- ✅ lz-string.min.js
|   |   |   |-- ✅ reference/
|   |   |   |   \-- ✅ lz-string-1.0.2.js
|   |   |   |-- ✅ tests/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   \-- ✅ jasmine-1.3.1/
|   |   |   |   |       |-- ✅ jasmine-html.js
|   |   |   |   |       |-- ✅ jasmine.css
|   |   |   |   |       |-- ✅ jasmine.js
|   |   |   |   |       \-- ✅ MIT.LICENSE
|   |   |   |   |-- ✅ lz-string-spec.js
|   |   |   |   \-- ✅ SpecRunner.html
|   |   |   |-- ✅ typings/
|   |   |   |   \-- ✅ lz-string.d.ts
|   |   |   |-- ✅ bower.json
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ magic-string/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ magic-string.cjs.d.ts
|   |   |   |   |-- ✅ magic-string.cjs.js
|   |   |   |   |-- ✅ magic-string.cjs.js.map
|   |   |   |   |-- ✅ magic-string.es.d.mts
|   |   |   |   |-- ✅ magic-string.es.mjs
|   |   |   |   |-- ✅ magic-string.es.mjs.map
|   |   |   |   |-- ✅ magic-string.umd.js
|   |   |   |   \-- ✅ magic-string.umd.js.map
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ math-intrinsics/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ constants/
|   |   |   |   |-- ✅ maxArrayLength.d.ts
|   |   |   |   |-- ✅ maxArrayLength.js
|   |   |   |   |-- ✅ maxSafeInteger.d.ts
|   |   |   |   |-- ✅ maxSafeInteger.js
|   |   |   |   |-- ✅ maxValue.d.ts
|   |   |   |   \-- ✅ maxValue.js
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ abs.d.ts
|   |   |   |-- ✅ abs.js
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ floor.d.ts
|   |   |   |-- ✅ floor.js
|   |   |   |-- ✅ isFinite.d.ts
|   |   |   |-- ✅ isFinite.js
|   |   |   |-- ✅ isInteger.d.ts
|   |   |   |-- ✅ isInteger.js
|   |   |   |-- ✅ isNaN.d.ts
|   |   |   |-- ✅ isNaN.js
|   |   |   |-- ✅ isNegativeZero.d.ts
|   |   |   |-- ✅ isNegativeZero.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ max.d.ts
|   |   |   |-- ✅ max.js
|   |   |   |-- ✅ min.d.ts
|   |   |   |-- ✅ min.js
|   |   |   |-- ✅ mod.d.ts
|   |   |   |-- ✅ mod.js
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ pow.d.ts
|   |   |   |-- ✅ pow.js
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ round.d.ts
|   |   |   |-- ✅ round.js
|   |   |   |-- ✅ sign.d.ts
|   |   |   |-- ✅ sign.js
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ merge2/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ micromatch/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ mime-db/
|   |   |   |-- ✅ db.json
|   |   |   |-- ✅ HISTORY.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ mime-types/
|   |   |   |-- ✅ HISTORY.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ min-indent/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ minimatch/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ commonjs/
|   |   |   |   |   |-- ✅ assert-valid-pattern.d.ts
|   |   |   |   |   |-- ✅ assert-valid-pattern.d.ts.map
|   |   |   |   |   |-- ✅ assert-valid-pattern.js
|   |   |   |   |   |-- ✅ assert-valid-pattern.js.map
|   |   |   |   |   |-- ✅ ast.d.ts
|   |   |   |   |   |-- ✅ ast.d.ts.map
|   |   |   |   |   |-- ✅ ast.js
|   |   |   |   |   |-- ✅ ast.js.map
|   |   |   |   |   |-- ✅ brace-expressions.d.ts
|   |   |   |   |   |-- ✅ brace-expressions.d.ts.map
|   |   |   |   |   |-- ✅ brace-expressions.js
|   |   |   |   |   |-- ✅ brace-expressions.js.map
|   |   |   |   |   |-- ✅ escape.d.ts
|   |   |   |   |   |-- ✅ escape.d.ts.map
|   |   |   |   |   |-- ✅ escape.js
|   |   |   |   |   |-- ✅ escape.js.map
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |-- ✅ unescape.d.ts
|   |   |   |   |   |-- ✅ unescape.d.ts.map
|   |   |   |   |   |-- ✅ unescape.js
|   |   |   |   |   \-- ✅ unescape.js.map
|   |   |   |   \-- ✅ esm/
|   |   |   |       |-- ✅ assert-valid-pattern.d.ts
|   |   |   |       |-- ✅ assert-valid-pattern.d.ts.map
|   |   |   |       |-- ✅ assert-valid-pattern.js
|   |   |   |       |-- ✅ assert-valid-pattern.js.map
|   |   |   |       |-- ✅ ast.d.ts
|   |   |   |       |-- ✅ ast.d.ts.map
|   |   |   |       |-- ✅ ast.js
|   |   |   |       |-- ✅ ast.js.map
|   |   |   |       |-- ✅ brace-expressions.d.ts
|   |   |   |       |-- ✅ brace-expressions.d.ts.map
|   |   |   |       |-- ✅ brace-expressions.js
|   |   |   |       |-- ✅ brace-expressions.js.map
|   |   |   |       |-- ✅ escape.d.ts
|   |   |   |       |-- ✅ escape.d.ts.map
|   |   |   |       |-- ✅ escape.js
|   |   |   |       |-- ✅ escape.js.map
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.d.ts.map
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ index.js.map
|   |   |   |       |-- ✅ package.json
|   |   |   |       |-- ✅ unescape.d.ts
|   |   |   |       |-- ✅ unescape.d.ts.map
|   |   |   |       |-- ✅ unescape.js
|   |   |   |       \-- ✅ unescape.js.map
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ minipass/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ commonjs/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   \-- ✅ package.json
|   |   |   |   \-- ✅ esm/
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.d.ts.map
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ index.js.map
|   |   |   |       \-- ✅ package.json
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ moment/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ locale/
|   |   |   |   |   |-- ✅ af.js
|   |   |   |   |   |-- ✅ ar-dz.js
|   |   |   |   |   |-- ✅ ar-kw.js
|   |   |   |   |   |-- ✅ ar-ly.js
|   |   |   |   |   |-- ✅ ar-ma.js
|   |   |   |   |   |-- ✅ ar-ps.js
|   |   |   |   |   |-- ✅ ar-sa.js
|   |   |   |   |   |-- ✅ ar-tn.js
|   |   |   |   |   |-- ✅ ar.js
|   |   |   |   |   |-- ✅ az.js
|   |   |   |   |   |-- ✅ be.js
|   |   |   |   |   |-- ✅ bg.js
|   |   |   |   |   |-- ✅ bm.js
|   |   |   |   |   |-- ✅ bn-bd.js
|   |   |   |   |   |-- ✅ bn.js
|   |   |   |   |   |-- ✅ bo.js
|   |   |   |   |   |-- ✅ br.js
|   |   |   |   |   |-- ✅ bs.js
|   |   |   |   |   |-- ✅ ca.js
|   |   |   |   |   |-- ✅ cs.js
|   |   |   |   |   |-- ✅ cv.js
|   |   |   |   |   |-- ✅ cy.js
|   |   |   |   |   |-- ✅ da.js
|   |   |   |   |   |-- ✅ de-at.js
|   |   |   |   |   |-- ✅ de-ch.js
|   |   |   |   |   |-- ✅ de.js
|   |   |   |   |   |-- ✅ dv.js
|   |   |   |   |   |-- ✅ el.js
|   |   |   |   |   |-- ✅ en-au.js
|   |   |   |   |   |-- ✅ en-ca.js
|   |   |   |   |   |-- ✅ en-gb.js
|   |   |   |   |   |-- ✅ en-ie.js
|   |   |   |   |   |-- ✅ en-il.js
|   |   |   |   |   |-- ✅ en-in.js
|   |   |   |   |   |-- ✅ en-nz.js
|   |   |   |   |   |-- ✅ en-sg.js
|   |   |   |   |   |-- ✅ eo.js
|   |   |   |   |   |-- ✅ es-do.js
|   |   |   |   |   |-- ✅ es-mx.js
|   |   |   |   |   |-- ✅ es-us.js
|   |   |   |   |   |-- ✅ es.js
|   |   |   |   |   |-- ✅ et.js
|   |   |   |   |   |-- ✅ eu.js
|   |   |   |   |   |-- ✅ fa.js
|   |   |   |   |   |-- ✅ fi.js
|   |   |   |   |   |-- ✅ fil.js
|   |   |   |   |   |-- ✅ fo.js
|   |   |   |   |   |-- ✅ fr-ca.js
|   |   |   |   |   |-- ✅ fr-ch.js
|   |   |   |   |   |-- ✅ fr.js
|   |   |   |   |   |-- ✅ fy.js
|   |   |   |   |   |-- ✅ ga.js
|   |   |   |   |   |-- ✅ gd.js
|   |   |   |   |   |-- ✅ gl.js
|   |   |   |   |   |-- ✅ gom-deva.js
|   |   |   |   |   |-- ✅ gom-latn.js
|   |   |   |   |   |-- ✅ gu.js
|   |   |   |   |   |-- ✅ he.js
|   |   |   |   |   |-- ✅ hi.js
|   |   |   |   |   |-- ✅ hr.js
|   |   |   |   |   |-- ✅ hu.js
|   |   |   |   |   |-- ✅ hy-am.js
|   |   |   |   |   |-- ✅ id.js
|   |   |   |   |   |-- ✅ is.js
|   |   |   |   |   |-- ✅ it-ch.js
|   |   |   |   |   |-- ✅ it.js
|   |   |   |   |   |-- ✅ ja.js
|   |   |   |   |   |-- ✅ jv.js
|   |   |   |   |   |-- ✅ ka.js
|   |   |   |   |   |-- ✅ kk.js
|   |   |   |   |   |-- ✅ km.js
|   |   |   |   |   |-- ✅ kn.js
|   |   |   |   |   |-- ✅ ko.js
|   |   |   |   |   |-- ✅ ku-kmr.js
|   |   |   |   |   |-- ✅ ku.js
|   |   |   |   |   |-- ✅ ky.js
|   |   |   |   |   |-- ✅ lb.js
|   |   |   |   |   |-- ✅ lo.js
|   |   |   |   |   |-- ✅ lt.js
|   |   |   |   |   |-- ✅ lv.js
|   |   |   |   |   |-- ✅ me.js
|   |   |   |   |   |-- ✅ mi.js
|   |   |   |   |   |-- ✅ mk.js
|   |   |   |   |   |-- ✅ ml.js
|   |   |   |   |   |-- ✅ mn.js
|   |   |   |   |   |-- ✅ mr.js
|   |   |   |   |   |-- ✅ ms-my.js
|   |   |   |   |   |-- ✅ ms.js
|   |   |   |   |   |-- ✅ mt.js
|   |   |   |   |   |-- ✅ my.js
|   |   |   |   |   |-- ✅ nb.js
|   |   |   |   |   |-- ✅ ne.js
|   |   |   |   |   |-- ✅ nl-be.js
|   |   |   |   |   |-- ✅ nl.js
|   |   |   |   |   |-- ✅ nn.js
|   |   |   |   |   |-- ✅ oc-lnc.js
|   |   |   |   |   |-- ✅ pa-in.js
|   |   |   |   |   |-- ✅ pl.js
|   |   |   |   |   |-- ✅ pt-br.js
|   |   |   |   |   |-- ✅ pt.js
|   |   |   |   |   |-- ✅ ro.js
|   |   |   |   |   |-- ✅ ru.js
|   |   |   |   |   |-- ✅ sd.js
|   |   |   |   |   |-- ✅ se.js
|   |   |   |   |   |-- ✅ si.js
|   |   |   |   |   |-- ✅ sk.js
|   |   |   |   |   |-- ✅ sl.js
|   |   |   |   |   |-- ✅ sq.js
|   |   |   |   |   |-- ✅ sr-cyrl.js
|   |   |   |   |   |-- ✅ sr.js
|   |   |   |   |   |-- ✅ ss.js
|   |   |   |   |   |-- ✅ sv.js
|   |   |   |   |   |-- ✅ sw.js
|   |   |   |   |   |-- ✅ ta.js
|   |   |   |   |   |-- ✅ te.js
|   |   |   |   |   |-- ✅ tet.js
|   |   |   |   |   |-- ✅ tg.js
|   |   |   |   |   |-- ✅ th.js
|   |   |   |   |   |-- ✅ tk.js
|   |   |   |   |   |-- ✅ tl-ph.js
|   |   |   |   |   |-- ✅ tlh.js
|   |   |   |   |   |-- ✅ tr.js
|   |   |   |   |   |-- ✅ tzl.js
|   |   |   |   |   |-- ✅ tzm-latn.js
|   |   |   |   |   |-- ✅ tzm.js
|   |   |   |   |   |-- ✅ ug-cn.js
|   |   |   |   |   |-- ✅ uk.js
|   |   |   |   |   |-- ✅ ur.js
|   |   |   |   |   |-- ✅ uz-latn.js
|   |   |   |   |   |-- ✅ uz.js
|   |   |   |   |   |-- ✅ vi.js
|   |   |   |   |   |-- ✅ x-pseudo.js
|   |   |   |   |   |-- ✅ yo.js
|   |   |   |   |   |-- ✅ zh-cn.js
|   |   |   |   |   |-- ✅ zh-hk.js
|   |   |   |   |   |-- ✅ zh-mo.js
|   |   |   |   |   \-- ✅ zh-tw.js
|   |   |   |   \-- ✅ moment.js
|   |   |   |-- ✅ locale/
|   |   |   |   |-- ✅ af.js
|   |   |   |   |-- ✅ ar-dz.js
|   |   |   |   |-- ✅ ar-kw.js
|   |   |   |   |-- ✅ ar-ly.js
|   |   |   |   |-- ✅ ar-ma.js
|   |   |   |   |-- ✅ ar-ps.js
|   |   |   |   |-- ✅ ar-sa.js
|   |   |   |   |-- ✅ ar-tn.js
|   |   |   |   |-- ✅ ar.js
|   |   |   |   |-- ✅ az.js
|   |   |   |   |-- ✅ be.js
|   |   |   |   |-- ✅ bg.js
|   |   |   |   |-- ✅ bm.js
|   |   |   |   |-- ✅ bn-bd.js
|   |   |   |   |-- ✅ bn.js
|   |   |   |   |-- ✅ bo.js
|   |   |   |   |-- ✅ br.js
|   |   |   |   |-- ✅ bs.js
|   |   |   |   |-- ✅ ca.js
|   |   |   |   |-- ✅ cs.js
|   |   |   |   |-- ✅ cv.js
|   |   |   |   |-- ✅ cy.js
|   |   |   |   |-- ✅ da.js
|   |   |   |   |-- ✅ de-at.js
|   |   |   |   |-- ✅ de-ch.js
|   |   |   |   |-- ✅ de.js
|   |   |   |   |-- ✅ dv.js
|   |   |   |   |-- ✅ el.js
|   |   |   |   |-- ✅ en-au.js
|   |   |   |   |-- ✅ en-ca.js
|   |   |   |   |-- ✅ en-gb.js
|   |   |   |   |-- ✅ en-ie.js
|   |   |   |   |-- ✅ en-il.js
|   |   |   |   |-- ✅ en-in.js
|   |   |   |   |-- ✅ en-nz.js
|   |   |   |   |-- ✅ en-sg.js
|   |   |   |   |-- ✅ eo.js
|   |   |   |   |-- ✅ es-do.js
|   |   |   |   |-- ✅ es-mx.js
|   |   |   |   |-- ✅ es-us.js
|   |   |   |   |-- ✅ es.js
|   |   |   |   |-- ✅ et.js
|   |   |   |   |-- ✅ eu.js
|   |   |   |   |-- ✅ fa.js
|   |   |   |   |-- ✅ fi.js
|   |   |   |   |-- ✅ fil.js
|   |   |   |   |-- ✅ fo.js
|   |   |   |   |-- ✅ fr-ca.js
|   |   |   |   |-- ✅ fr-ch.js
|   |   |   |   |-- ✅ fr.js
|   |   |   |   |-- ✅ fy.js
|   |   |   |   |-- ✅ ga.js
|   |   |   |   |-- ✅ gd.js
|   |   |   |   |-- ✅ gl.js
|   |   |   |   |-- ✅ gom-deva.js
|   |   |   |   |-- ✅ gom-latn.js
|   |   |   |   |-- ✅ gu.js
|   |   |   |   |-- ✅ he.js
|   |   |   |   |-- ✅ hi.js
|   |   |   |   |-- ✅ hr.js
|   |   |   |   |-- ✅ hu.js
|   |   |   |   |-- ✅ hy-am.js
|   |   |   |   |-- ✅ id.js
|   |   |   |   |-- ✅ is.js
|   |   |   |   |-- ✅ it-ch.js
|   |   |   |   |-- ✅ it.js
|   |   |   |   |-- ✅ ja.js
|   |   |   |   |-- ✅ jv.js
|   |   |   |   |-- ✅ ka.js
|   |   |   |   |-- ✅ kk.js
|   |   |   |   |-- ✅ km.js
|   |   |   |   |-- ✅ kn.js
|   |   |   |   |-- ✅ ko.js
|   |   |   |   |-- ✅ ku-kmr.js
|   |   |   |   |-- ✅ ku.js
|   |   |   |   |-- ✅ ky.js
|   |   |   |   |-- ✅ lb.js
|   |   |   |   |-- ✅ lo.js
|   |   |   |   |-- ✅ lt.js
|   |   |   |   |-- ✅ lv.js
|   |   |   |   |-- ✅ me.js
|   |   |   |   |-- ✅ mi.js
|   |   |   |   |-- ✅ mk.js
|   |   |   |   |-- ✅ ml.js
|   |   |   |   |-- ✅ mn.js
|   |   |   |   |-- ✅ mr.js
|   |   |   |   |-- ✅ ms-my.js
|   |   |   |   |-- ✅ ms.js
|   |   |   |   |-- ✅ mt.js
|   |   |   |   |-- ✅ my.js
|   |   |   |   |-- ✅ nb.js
|   |   |   |   |-- ✅ ne.js
|   |   |   |   |-- ✅ nl-be.js
|   |   |   |   |-- ✅ nl.js
|   |   |   |   |-- ✅ nn.js
|   |   |   |   |-- ✅ oc-lnc.js
|   |   |   |   |-- ✅ pa-in.js
|   |   |   |   |-- ✅ pl.js
|   |   |   |   |-- ✅ pt-br.js
|   |   |   |   |-- ✅ pt.js
|   |   |   |   |-- ✅ ro.js
|   |   |   |   |-- ✅ ru.js
|   |   |   |   |-- ✅ sd.js
|   |   |   |   |-- ✅ se.js
|   |   |   |   |-- ✅ si.js
|   |   |   |   |-- ✅ sk.js
|   |   |   |   |-- ✅ sl.js
|   |   |   |   |-- ✅ sq.js
|   |   |   |   |-- ✅ sr-cyrl.js
|   |   |   |   |-- ✅ sr.js
|   |   |   |   |-- ✅ ss.js
|   |   |   |   |-- ✅ sv.js
|   |   |   |   |-- ✅ sw.js
|   |   |   |   |-- ✅ ta.js
|   |   |   |   |-- ✅ te.js
|   |   |   |   |-- ✅ tet.js
|   |   |   |   |-- ✅ tg.js
|   |   |   |   |-- ✅ th.js
|   |   |   |   |-- ✅ tk.js
|   |   |   |   |-- ✅ tl-ph.js
|   |   |   |   |-- ✅ tlh.js
|   |   |   |   |-- ✅ tr.js
|   |   |   |   |-- ✅ tzl.js
|   |   |   |   |-- ✅ tzm-latn.js
|   |   |   |   |-- ✅ tzm.js
|   |   |   |   |-- ✅ ug-cn.js
|   |   |   |   |-- ✅ uk.js
|   |   |   |   |-- ✅ ur.js
|   |   |   |   |-- ✅ uz-latn.js
|   |   |   |   |-- ✅ uz.js
|   |   |   |   |-- ✅ vi.js
|   |   |   |   |-- ✅ x-pseudo.js
|   |   |   |   |-- ✅ yo.js
|   |   |   |   |-- ✅ zh-cn.js
|   |   |   |   |-- ✅ zh-hk.js
|   |   |   |   |-- ✅ zh-mo.js
|   |   |   |   \-- ✅ zh-tw.js
|   |   |   |-- ✅ min/
|   |   |   |   |-- ✅ locales.js
|   |   |   |   |-- ✅ locales.min.js
|   |   |   |   |-- ✅ locales.min.js.map
|   |   |   |   |-- ✅ moment-with-locales.js
|   |   |   |   |-- ✅ moment-with-locales.min.js
|   |   |   |   |-- ✅ moment-with-locales.min.js.map
|   |   |   |   |-- ✅ moment.min.js
|   |   |   |   \-- ✅ moment.min.js.map
|   |   |   |-- ✅ src/
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ create/
|   |   |   |   |   |   |-- ✅ check-overflow.js
|   |   |   |   |   |   |-- ✅ date-from-array.js
|   |   |   |   |   |   |-- ✅ from-anything.js
|   |   |   |   |   |   |-- ✅ from-array.js
|   |   |   |   |   |   |-- ✅ from-object.js
|   |   |   |   |   |   |-- ✅ from-string-and-array.js
|   |   |   |   |   |   |-- ✅ from-string-and-format.js
|   |   |   |   |   |   |-- ✅ from-string.js
|   |   |   |   |   |   |-- ✅ local.js
|   |   |   |   |   |   |-- ✅ parsing-flags.js
|   |   |   |   |   |   |-- ✅ utc.js
|   |   |   |   |   |   \-- ✅ valid.js
|   |   |   |   |   |-- ✅ duration/
|   |   |   |   |   |   |-- ✅ abs.js
|   |   |   |   |   |   |-- ✅ add-subtract.js
|   |   |   |   |   |   |-- ✅ as.js
|   |   |   |   |   |   |-- ✅ bubble.js
|   |   |   |   |   |   |-- ✅ clone.js
|   |   |   |   |   |   |-- ✅ constructor.js
|   |   |   |   |   |   |-- ✅ create.js
|   |   |   |   |   |   |-- ✅ duration.js
|   |   |   |   |   |   |-- ✅ get.js
|   |   |   |   |   |   |-- ✅ humanize.js
|   |   |   |   |   |   |-- ✅ iso-string.js
|   |   |   |   |   |   |-- ✅ prototype.js
|   |   |   |   |   |   \-- ✅ valid.js
|   |   |   |   |   |-- ✅ format/
|   |   |   |   |   |   \-- ✅ format.js
|   |   |   |   |   |-- ✅ locale/
|   |   |   |   |   |   |-- ✅ base-config.js
|   |   |   |   |   |   |-- ✅ calendar.js
|   |   |   |   |   |   |-- ✅ constructor.js
|   |   |   |   |   |   |-- ✅ en.js
|   |   |   |   |   |   |-- ✅ formats.js
|   |   |   |   |   |   |-- ✅ invalid.js
|   |   |   |   |   |   |-- ✅ lists.js
|   |   |   |   |   |   |-- ✅ locale.js
|   |   |   |   |   |   |-- ✅ locales.js
|   |   |   |   |   |   |-- ✅ ordinal.js
|   |   |   |   |   |   |-- ✅ pre-post-format.js
|   |   |   |   |   |   |-- ✅ prototype.js
|   |   |   |   |   |   |-- ✅ relative.js
|   |   |   |   |   |   \-- ✅ set.js
|   |   |   |   |   |-- ✅ moment/
|   |   |   |   |   |   |-- ✅ add-subtract.js
|   |   |   |   |   |   |-- ✅ calendar.js
|   |   |   |   |   |   |-- ✅ clone.js
|   |   |   |   |   |   |-- ✅ compare.js
|   |   |   |   |   |   |-- ✅ constructor.js
|   |   |   |   |   |   |-- ✅ creation-data.js
|   |   |   |   |   |   |-- ✅ diff.js
|   |   |   |   |   |   |-- ✅ format.js
|   |   |   |   |   |   |-- ✅ from.js
|   |   |   |   |   |   |-- ✅ get-set.js
|   |   |   |   |   |   |-- ✅ locale.js
|   |   |   |   |   |   |-- ✅ min-max.js
|   |   |   |   |   |   |-- ✅ moment.js
|   |   |   |   |   |   |-- ✅ now.js
|   |   |   |   |   |   |-- ✅ prototype.js
|   |   |   |   |   |   |-- ✅ start-end-of.js
|   |   |   |   |   |   |-- ✅ to-type.js
|   |   |   |   |   |   |-- ✅ to.js
|   |   |   |   |   |   \-- ✅ valid.js
|   |   |   |   |   |-- ✅ parse/
|   |   |   |   |   |   |-- ✅ regex.js
|   |   |   |   |   |   \-- ✅ token.js
|   |   |   |   |   |-- ✅ units/
|   |   |   |   |   |   |-- ✅ aliases.js
|   |   |   |   |   |   |-- ✅ constants.js
|   |   |   |   |   |   |-- ✅ day-of-month.js
|   |   |   |   |   |   |-- ✅ day-of-week.js
|   |   |   |   |   |   |-- ✅ day-of-year.js
|   |   |   |   |   |   |-- ✅ era.js
|   |   |   |   |   |   |-- ✅ hour.js
|   |   |   |   |   |   |-- ✅ millisecond.js
|   |   |   |   |   |   |-- ✅ minute.js
|   |   |   |   |   |   |-- ✅ month.js
|   |   |   |   |   |   |-- ✅ offset.js
|   |   |   |   |   |   |-- ✅ priorities.js
|   |   |   |   |   |   |-- ✅ quarter.js
|   |   |   |   |   |   |-- ✅ second.js
|   |   |   |   |   |   |-- ✅ timestamp.js
|   |   |   |   |   |   |-- ✅ timezone.js
|   |   |   |   |   |   |-- ✅ units.js
|   |   |   |   |   |   |-- ✅ week-calendar-utils.js
|   |   |   |   |   |   |-- ✅ week-year.js
|   |   |   |   |   |   |-- ✅ week.js
|   |   |   |   |   |   \-- ✅ year.js
|   |   |   |   |   \-- ✅ utils/
|   |   |   |   |       |-- ✅ abs-ceil.js
|   |   |   |   |       |-- ✅ abs-floor.js
|   |   |   |   |       |-- ✅ abs-round.js
|   |   |   |   |       |-- ✅ compare-arrays.js
|   |   |   |   |       |-- ✅ defaults.js
|   |   |   |   |       |-- ✅ deprecate.js
|   |   |   |   |       |-- ✅ extend.js
|   |   |   |   |       |-- ✅ has-own-prop.js
|   |   |   |   |       |-- ✅ hooks.js
|   |   |   |   |       |-- ✅ index-of.js
|   |   |   |   |       |-- ✅ is-array.js
|   |   |   |   |       |-- ✅ is-calendar-spec.js
|   |   |   |   |       |-- ✅ is-date.js
|   |   |   |   |       |-- ✅ is-function.js
|   |   |   |   |       |-- ✅ is-leap-year.js
|   |   |   |   |       |-- ✅ is-moment-input.js
|   |   |   |   |       |-- ✅ is-number.js
|   |   |   |   |       |-- ✅ is-object-empty.js
|   |   |   |   |       |-- ✅ is-object.js
|   |   |   |   |       |-- ✅ is-string.js
|   |   |   |   |       |-- ✅ is-undefined.js
|   |   |   |   |       |-- ✅ keys.js
|   |   |   |   |       |-- ✅ map.js
|   |   |   |   |       |-- ✅ mod.js
|   |   |   |   |       |-- ✅ some.js
|   |   |   |   |       |-- ✅ to-int.js
|   |   |   |   |       \-- ✅ zero-fill.js
|   |   |   |   |-- ✅ locale/
|   |   |   |   |   |-- ✅ af.js
|   |   |   |   |   |-- ✅ ar-dz.js
|   |   |   |   |   |-- ✅ ar-kw.js
|   |   |   |   |   |-- ✅ ar-ly.js
|   |   |   |   |   |-- ✅ ar-ma.js
|   |   |   |   |   |-- ✅ ar-ps.js
|   |   |   |   |   |-- ✅ ar-sa.js
|   |   |   |   |   |-- ✅ ar-tn.js
|   |   |   |   |   |-- ✅ ar.js
|   |   |   |   |   |-- ✅ az.js
|   |   |   |   |   |-- ✅ be.js
|   |   |   |   |   |-- ✅ bg.js
|   |   |   |   |   |-- ✅ bm.js
|   |   |   |   |   |-- ✅ bn-bd.js
|   |   |   |   |   |-- ✅ bn.js
|   |   |   |   |   |-- ✅ bo.js
|   |   |   |   |   |-- ✅ br.js
|   |   |   |   |   |-- ✅ bs.js
|   |   |   |   |   |-- ✅ ca.js
|   |   |   |   |   |-- ✅ cs.js
|   |   |   |   |   |-- ✅ cv.js
|   |   |   |   |   |-- ✅ cy.js
|   |   |   |   |   |-- ✅ da.js
|   |   |   |   |   |-- ✅ de-at.js
|   |   |   |   |   |-- ✅ de-ch.js
|   |   |   |   |   |-- ✅ de.js
|   |   |   |   |   |-- ✅ dv.js
|   |   |   |   |   |-- ✅ el.js
|   |   |   |   |   |-- ✅ en-au.js
|   |   |   |   |   |-- ✅ en-ca.js
|   |   |   |   |   |-- ✅ en-gb.js
|   |   |   |   |   |-- ✅ en-ie.js
|   |   |   |   |   |-- ✅ en-il.js
|   |   |   |   |   |-- ✅ en-in.js
|   |   |   |   |   |-- ✅ en-nz.js
|   |   |   |   |   |-- ✅ en-sg.js
|   |   |   |   |   |-- ✅ eo.js
|   |   |   |   |   |-- ✅ es-do.js
|   |   |   |   |   |-- ✅ es-mx.js
|   |   |   |   |   |-- ✅ es-us.js
|   |   |   |   |   |-- ✅ es.js
|   |   |   |   |   |-- ✅ et.js
|   |   |   |   |   |-- ✅ eu.js
|   |   |   |   |   |-- ✅ fa.js
|   |   |   |   |   |-- ✅ fi.js
|   |   |   |   |   |-- ✅ fil.js
|   |   |   |   |   |-- ✅ fo.js
|   |   |   |   |   |-- ✅ fr-ca.js
|   |   |   |   |   |-- ✅ fr-ch.js
|   |   |   |   |   |-- ✅ fr.js
|   |   |   |   |   |-- ✅ fy.js
|   |   |   |   |   |-- ✅ ga.js
|   |   |   |   |   |-- ✅ gd.js
|   |   |   |   |   |-- ✅ gl.js
|   |   |   |   |   |-- ✅ gom-deva.js
|   |   |   |   |   |-- ✅ gom-latn.js
|   |   |   |   |   |-- ✅ gu.js
|   |   |   |   |   |-- ✅ he.js
|   |   |   |   |   |-- ✅ hi.js
|   |   |   |   |   |-- ✅ hr.js
|   |   |   |   |   |-- ✅ hu.js
|   |   |   |   |   |-- ✅ hy-am.js
|   |   |   |   |   |-- ✅ id.js
|   |   |   |   |   |-- ✅ is.js
|   |   |   |   |   |-- ✅ it-ch.js
|   |   |   |   |   |-- ✅ it.js
|   |   |   |   |   |-- ✅ ja.js
|   |   |   |   |   |-- ✅ jv.js
|   |   |   |   |   |-- ✅ ka.js
|   |   |   |   |   |-- ✅ kk.js
|   |   |   |   |   |-- ✅ km.js
|   |   |   |   |   |-- ✅ kn.js
|   |   |   |   |   |-- ✅ ko.js
|   |   |   |   |   |-- ✅ ku-kmr.js
|   |   |   |   |   |-- ✅ ku.js
|   |   |   |   |   |-- ✅ ky.js
|   |   |   |   |   |-- ✅ lb.js
|   |   |   |   |   |-- ✅ lo.js
|   |   |   |   |   |-- ✅ lt.js
|   |   |   |   |   |-- ✅ lv.js
|   |   |   |   |   |-- ✅ me.js
|   |   |   |   |   |-- ✅ mi.js
|   |   |   |   |   |-- ✅ mk.js
|   |   |   |   |   |-- ✅ ml.js
|   |   |   |   |   |-- ✅ mn.js
|   |   |   |   |   |-- ✅ mr.js
|   |   |   |   |   |-- ✅ ms-my.js
|   |   |   |   |   |-- ✅ ms.js
|   |   |   |   |   |-- ✅ mt.js
|   |   |   |   |   |-- ✅ my.js
|   |   |   |   |   |-- ✅ nb.js
|   |   |   |   |   |-- ✅ ne.js
|   |   |   |   |   |-- ✅ nl-be.js
|   |   |   |   |   |-- ✅ nl.js
|   |   |   |   |   |-- ✅ nn.js
|   |   |   |   |   |-- ✅ oc-lnc.js
|   |   |   |   |   |-- ✅ pa-in.js
|   |   |   |   |   |-- ✅ pl.js
|   |   |   |   |   |-- ✅ pt-br.js
|   |   |   |   |   |-- ✅ pt.js
|   |   |   |   |   |-- ✅ ro.js
|   |   |   |   |   |-- ✅ ru.js
|   |   |   |   |   |-- ✅ sd.js
|   |   |   |   |   |-- ✅ se.js
|   |   |   |   |   |-- ✅ si.js
|   |   |   |   |   |-- ✅ sk.js
|   |   |   |   |   |-- ✅ sl.js
|   |   |   |   |   |-- ✅ sq.js
|   |   |   |   |   |-- ✅ sr-cyrl.js
|   |   |   |   |   |-- ✅ sr.js
|   |   |   |   |   |-- ✅ ss.js
|   |   |   |   |   |-- ✅ sv.js
|   |   |   |   |   |-- ✅ sw.js
|   |   |   |   |   |-- ✅ ta.js
|   |   |   |   |   |-- ✅ te.js
|   |   |   |   |   |-- ✅ tet.js
|   |   |   |   |   |-- ✅ tg.js
|   |   |   |   |   |-- ✅ th.js
|   |   |   |   |   |-- ✅ tk.js
|   |   |   |   |   |-- ✅ tl-ph.js
|   |   |   |   |   |-- ✅ tlh.js
|   |   |   |   |   |-- ✅ tr.js
|   |   |   |   |   |-- ✅ tzl.js
|   |   |   |   |   |-- ✅ tzm-latn.js
|   |   |   |   |   |-- ✅ tzm.js
|   |   |   |   |   |-- ✅ ug-cn.js
|   |   |   |   |   |-- ✅ uk.js
|   |   |   |   |   |-- ✅ ur.js
|   |   |   |   |   |-- ✅ uz-latn.js
|   |   |   |   |   |-- ✅ uz.js
|   |   |   |   |   |-- ✅ vi.js
|   |   |   |   |   |-- ✅ x-pseudo.js
|   |   |   |   |   |-- ✅ yo.js
|   |   |   |   |   |-- ✅ zh-cn.js
|   |   |   |   |   |-- ✅ zh-hk.js
|   |   |   |   |   |-- ✅ zh-mo.js
|   |   |   |   |   \-- ✅ zh-tw.js
|   |   |   |   \-- ✅ moment.js
|   |   |   |-- ✅ ts3.1-typings/
|   |   |   |   \-- ✅ moment.d.ts
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ ender.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ moment.d.ts
|   |   |   |-- ✅ moment.js
|   |   |   |-- ✅ package.js
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ ms/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ mz/
|   |   |   |-- ✅ child_process.js
|   |   |   |-- ✅ crypto.js
|   |   |   |-- ✅ dns.js
|   |   |   |-- ✅ fs.js
|   |   |   |-- ✅ HISTORY.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ readline.js
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ zlib.js
|   |   |-- ✅ nanoid/
|   |   |   |-- ✅ async/
|   |   |   |   |-- ✅ index.browser.cjs
|   |   |   |   |-- ✅ index.browser.js
|   |   |   |   |-- ✅ index.cjs
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ index.native.js
|   |   |   |   \-- ✅ package.json
|   |   |   |-- ✅ bin/
|   |   |   |   \-- ✅ nanoid.cjs
|   |   |   |-- ✅ non-secure/
|   |   |   |   |-- ✅ index.cjs
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ package.json
|   |   |   |-- ✅ url-alphabet/
|   |   |   |   |-- ✅ index.cjs
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ package.json
|   |   |   |-- ✅ index.browser.cjs
|   |   |   |-- ✅ index.browser.js
|   |   |   |-- ✅ index.cjs
|   |   |   |-- ✅ index.d.cts
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ nanoid.js
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ node-releases/
|   |   |   |-- ✅ data/
|   |   |   |   |-- ✅ processed/
|   |   |   |   |   \-- ✅ envs.json
|   |   |   |   \-- ✅ release-schedule/
|   |   |   |       \-- ✅ release-schedule.json
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ normalize-path/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ normalize-range/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ nwsapi/
|   |   |   |-- ✅ dist/
|   |   |   |   \-- ✅ lint.log
|   |   |   |-- ✅ src/
|   |   |   |   |-- ✅ modules/
|   |   |   |   |   |-- ✅ nwsapi-jquery.js
|   |   |   |   |   \-- ✅ nwsapi-traversal.js
|   |   |   |   \-- ✅ nwsapi.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ object-assign/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ object-hash/
|   |   |   |-- ✅ dist/
|   |   |   |   \-- ✅ object_hash.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.markdown
|   |   |-- ✅ object-inspect/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ example/
|   |   |   |   |-- ✅ all.js
|   |   |   |   |-- ✅ circular.js
|   |   |   |   |-- ✅ fn.js
|   |   |   |   \-- ✅ inspect.js
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ browser/
|   |   |   |   |   \-- ✅ dom.js
|   |   |   |   |-- ✅ bigint.js
|   |   |   |   |-- ✅ circular.js
|   |   |   |   |-- ✅ deep.js
|   |   |   |   |-- ✅ element.js
|   |   |   |   |-- ✅ err.js
|   |   |   |   |-- ✅ fakes.js
|   |   |   |   |-- ✅ fn.js
|   |   |   |   |-- ✅ global.js
|   |   |   |   |-- ✅ has.js
|   |   |   |   |-- ✅ holes.js
|   |   |   |   |-- ✅ indent-option.js
|   |   |   |   |-- ✅ inspect.js
|   |   |   |   |-- ✅ lowbyte.js
|   |   |   |   |-- ✅ number.js
|   |   |   |   |-- ✅ quoteStyle.js
|   |   |   |   |-- ✅ toStringTag.js
|   |   |   |   |-- ✅ undef.js
|   |   |   |   \-- ✅ values.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package-support.json
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ readme.markdown
|   |   |   |-- ✅ test-core-js.js
|   |   |   \-- ✅ util.inspect.js
|   |   |-- ✅ object-is/
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ implementation.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ shimmed.js
|   |   |   |   \-- ✅ tests.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ auto.js
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ implementation.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ polyfill.js
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ shim.js
|   |   |-- ✅ object-keys/
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .travis.yml
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ implementation.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ isArguments.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ object.assign/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ dist/
|   |   |   |   \-- ✅ browser.js
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ implementation.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ native.js
|   |   |   |   |-- ✅ ses-compat.js
|   |   |   |   |-- ✅ shimmed.js
|   |   |   |   \-- ✅ tests.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ auto.js
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ hasSymbols.js
|   |   |   |-- ✅ implementation.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ polyfill.js
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ shim.js
|   |   |-- ✅ package-json-from-dist/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ commonjs/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   \-- ✅ package.json
|   |   |   |   \-- ✅ esm/
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.d.ts.map
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ index.js.map
|   |   |   |       \-- ✅ package.json
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ parse5/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ cjs/
|   |   |   |   |   |-- ✅ common/
|   |   |   |   |   |   |-- ✅ doctype.d.ts
|   |   |   |   |   |   |-- ✅ doctype.js
|   |   |   |   |   |   |-- ✅ error-codes.d.ts
|   |   |   |   |   |   |-- ✅ error-codes.js
|   |   |   |   |   |   |-- ✅ foreign-content.d.ts
|   |   |   |   |   |   |-- ✅ foreign-content.js
|   |   |   |   |   |   |-- ✅ html.d.ts
|   |   |   |   |   |   |-- ✅ html.js
|   |   |   |   |   |   |-- ✅ token.d.ts
|   |   |   |   |   |   |-- ✅ token.js
|   |   |   |   |   |   |-- ✅ unicode.d.ts
|   |   |   |   |   |   \-- ✅ unicode.js
|   |   |   |   |   |-- ✅ parser/
|   |   |   |   |   |   |-- ✅ formatting-element-list.d.ts
|   |   |   |   |   |   |-- ✅ formatting-element-list.js
|   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ open-element-stack.d.ts
|   |   |   |   |   |   \-- ✅ open-element-stack.js
|   |   |   |   |   |-- ✅ serializer/
|   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   |-- ✅ tokenizer/
|   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ preprocessor.d.ts
|   |   |   |   |   |   \-- ✅ preprocessor.js
|   |   |   |   |   |-- ✅ tree-adapters/
|   |   |   |   |   |   |-- ✅ default.d.ts
|   |   |   |   |   |   |-- ✅ default.js
|   |   |   |   |   |   |-- ✅ interface.d.ts
|   |   |   |   |   |   \-- ✅ interface.js
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   \-- ✅ package.json
|   |   |   |   |-- ✅ common/
|   |   |   |   |   |-- ✅ doctype.d.ts
|   |   |   |   |   |-- ✅ doctype.js
|   |   |   |   |   |-- ✅ error-codes.d.ts
|   |   |   |   |   |-- ✅ error-codes.js
|   |   |   |   |   |-- ✅ foreign-content.d.ts
|   |   |   |   |   |-- ✅ foreign-content.js
|   |   |   |   |   |-- ✅ html.d.ts
|   |   |   |   |   |-- ✅ html.js
|   |   |   |   |   |-- ✅ token.d.ts
|   |   |   |   |   |-- ✅ token.js
|   |   |   |   |   |-- ✅ unicode.d.ts
|   |   |   |   |   \-- ✅ unicode.js
|   |   |   |   |-- ✅ parser/
|   |   |   |   |   |-- ✅ formatting-element-list.d.ts
|   |   |   |   |   |-- ✅ formatting-element-list.js
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ open-element-stack.d.ts
|   |   |   |   |   \-- ✅ open-element-stack.js
|   |   |   |   |-- ✅ serializer/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ tokenizer/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ preprocessor.d.ts
|   |   |   |   |   \-- ✅ preprocessor.js
|   |   |   |   |-- ✅ tree-adapters/
|   |   |   |   |   |-- ✅ default.d.ts
|   |   |   |   |   |-- ✅ default.js
|   |   |   |   |   |-- ✅ interface.d.ts
|   |   |   |   |   \-- ✅ interface.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ path-key/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ path-parse/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ path-scurry/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ commonjs/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   \-- ✅ package.json
|   |   |   |   \-- ✅ esm/
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.d.ts.map
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ index.js.map
|   |   |   |       \-- ✅ package.json
|   |   |   |-- ✅ node_modules/
|   |   |   |   \-- ✅ lru-cache/
|   |   |   |       |-- ✅ dist/
|   |   |   |       |   |-- ✅ commonjs/
|   |   |   |       |   |   |-- ✅ index.d.ts
|   |   |   |       |   |   |-- ✅ index.d.ts.map
|   |   |   |       |   |   |-- ✅ index.js
|   |   |   |       |   |   |-- ✅ index.js.map
|   |   |   |       |   |   |-- ✅ index.min.js
|   |   |   |       |   |   |-- ✅ index.min.js.map
|   |   |   |       |   |   \-- ✅ package.json
|   |   |   |       |   \-- ✅ esm/
|   |   |   |       |       |-- ✅ index.d.ts
|   |   |   |       |       |-- ✅ index.d.ts.map
|   |   |   |       |       |-- ✅ index.js
|   |   |   |       |       |-- ✅ index.js.map
|   |   |   |       |       |-- ✅ index.min.js
|   |   |   |       |       |-- ✅ index.min.js.map
|   |   |   |       |       \-- ✅ package.json
|   |   |   |       |-- ✅ LICENSE
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ README.md
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ pathe/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ shared/
|   |   |   |   |   |-- ✅ pathe.BSlhyZSM.cjs
|   |   |   |   |   \-- ✅ pathe.M-eThtNZ.mjs
|   |   |   |   |-- ✅ index.cjs
|   |   |   |   |-- ✅ index.d.cts
|   |   |   |   |-- ✅ index.d.mts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.mjs
|   |   |   |   |-- ✅ utils.cjs
|   |   |   |   |-- ✅ utils.d.cts
|   |   |   |   |-- ✅ utils.d.mts
|   |   |   |   |-- ✅ utils.d.ts
|   |   |   |   \-- ✅ utils.mjs
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ utils.d.ts
|   |   |-- ✅ pathval/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ picocolors/
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ picocolors.browser.js
|   |   |   |-- ✅ picocolors.d.ts
|   |   |   |-- ✅ picocolors.js
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ types.d.ts
|   |   |-- ✅ picomatch/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ constants.js
|   |   |   |   |-- ✅ parse.js
|   |   |   |   |-- ✅ picomatch.js
|   |   |   |   |-- ✅ scan.js
|   |   |   |   \-- ✅ utils.js
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ pify/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ pirates/
|   |   |   |-- ✅ lib/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ possible-typed-array-names/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ postcss/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ at-rule.d.ts
|   |   |   |   |-- ✅ at-rule.js
|   |   |   |   |-- ✅ comment.d.ts
|   |   |   |   |-- ✅ comment.js
|   |   |   |   |-- ✅ container.d.ts
|   |   |   |   |-- ✅ container.js
|   |   |   |   |-- ✅ css-syntax-error.d.ts
|   |   |   |   |-- ✅ css-syntax-error.js
|   |   |   |   |-- ✅ declaration.d.ts
|   |   |   |   |-- ✅ declaration.js
|   |   |   |   |-- ✅ document.d.ts
|   |   |   |   |-- ✅ document.js
|   |   |   |   |-- ✅ fromJSON.d.ts
|   |   |   |   |-- ✅ fromJSON.js
|   |   |   |   |-- ✅ input.d.ts
|   |   |   |   |-- ✅ input.js
|   |   |   |   |-- ✅ lazy-result.d.ts
|   |   |   |   |-- ✅ lazy-result.js
|   |   |   |   |-- ✅ list.d.ts
|   |   |   |   |-- ✅ list.js
|   |   |   |   |-- ✅ map-generator.js
|   |   |   |   |-- ✅ no-work-result.d.ts
|   |   |   |   |-- ✅ no-work-result.js
|   |   |   |   |-- ✅ node.d.ts
|   |   |   |   |-- ✅ node.js
|   |   |   |   |-- ✅ parse.d.ts
|   |   |   |   |-- ✅ parse.js
|   |   |   |   |-- ✅ parser.js
|   |   |   |   |-- ✅ postcss.d.mts
|   |   |   |   |-- ✅ postcss.d.ts
|   |   |   |   |-- ✅ postcss.js
|   |   |   |   |-- ✅ postcss.mjs
|   |   |   |   |-- ✅ previous-map.d.ts
|   |   |   |   |-- ✅ previous-map.js
|   |   |   |   |-- ✅ processor.d.ts
|   |   |   |   |-- ✅ processor.js
|   |   |   |   |-- ✅ result.d.ts
|   |   |   |   |-- ✅ result.js
|   |   |   |   |-- ✅ root.d.ts
|   |   |   |   |-- ✅ root.js
|   |   |   |   |-- ✅ rule.d.ts
|   |   |   |   |-- ✅ rule.js
|   |   |   |   |-- ✅ stringifier.d.ts
|   |   |   |   |-- ✅ stringifier.js
|   |   |   |   |-- ✅ stringify.d.ts
|   |   |   |   |-- ✅ stringify.js
|   |   |   |   |-- ✅ symbols.js
|   |   |   |   |-- ✅ terminal-highlight.js
|   |   |   |   |-- ✅ tokenize.js
|   |   |   |   |-- ✅ warn-once.js
|   |   |   |   |-- ✅ warning.d.ts
|   |   |   |   \-- ✅ warning.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ postcss-import/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ assign-layer-names.js
|   |   |   |   |-- ✅ data-url.js
|   |   |   |   |-- ✅ join-layer.js
|   |   |   |   |-- ✅ join-media.js
|   |   |   |   |-- ✅ load-content.js
|   |   |   |   |-- ✅ parse-statements.js
|   |   |   |   |-- ✅ process-content.js
|   |   |   |   \-- ✅ resolve-id.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ postcss-js/
|   |   |   |-- ✅ async.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ index.mjs
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ objectifier.js
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ parser.js
|   |   |   |-- ✅ process-result.js
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ sync.js
|   |   |-- ✅ postcss-load-config/
|   |   |   |-- ✅ src/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ options.js
|   |   |   |   |-- ✅ plugins.js
|   |   |   |   \-- ✅ req.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ postcss-nested/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ postcss-selector-parser/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ selectors/
|   |   |   |   |   |-- ✅ attribute.js
|   |   |   |   |   |-- ✅ className.js
|   |   |   |   |   |-- ✅ combinator.js
|   |   |   |   |   |-- ✅ comment.js
|   |   |   |   |   |-- ✅ constructors.js
|   |   |   |   |   |-- ✅ container.js
|   |   |   |   |   |-- ✅ guards.js
|   |   |   |   |   |-- ✅ id.js
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ namespace.js
|   |   |   |   |   |-- ✅ nesting.js
|   |   |   |   |   |-- ✅ node.js
|   |   |   |   |   |-- ✅ pseudo.js
|   |   |   |   |   |-- ✅ root.js
|   |   |   |   |   |-- ✅ selector.js
|   |   |   |   |   |-- ✅ string.js
|   |   |   |   |   |-- ✅ tag.js
|   |   |   |   |   |-- ✅ types.js
|   |   |   |   |   \-- ✅ universal.js
|   |   |   |   |-- ✅ util/
|   |   |   |   |   |-- ✅ ensureObject.js
|   |   |   |   |   |-- ✅ getProp.js
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ stripComments.js
|   |   |   |   |   \-- ✅ unesc.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ parser.js
|   |   |   |   |-- ✅ processor.js
|   |   |   |   |-- ✅ sortAscending.js
|   |   |   |   |-- ✅ tokenize.js
|   |   |   |   \-- ✅ tokenTypes.js
|   |   |   |-- ✅ API.md
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ LICENSE-MIT
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ postcss-selector-parser.d.ts
|   |   |   \-- ✅ README.md
|   |   |-- ✅ postcss-value-parser/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ parse.js
|   |   |   |   |-- ✅ stringify.js
|   |   |   |   |-- ✅ unit.js
|   |   |   |   \-- ✅ walk.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ pretty-format/
|   |   |   |-- ✅ build/
|   |   |   |   |-- ✅ plugins/
|   |   |   |   |   |-- ✅ lib/
|   |   |   |   |   |   |-- ✅ escapeHTML.d.ts
|   |   |   |   |   |   |-- ✅ escapeHTML.js
|   |   |   |   |   |   |-- ✅ markup.d.ts
|   |   |   |   |   |   \-- ✅ markup.js
|   |   |   |   |   |-- ✅ AsymmetricMatcher.d.ts
|   |   |   |   |   |-- ✅ AsymmetricMatcher.js
|   |   |   |   |   |-- ✅ ConvertAnsi.d.ts
|   |   |   |   |   |-- ✅ ConvertAnsi.js
|   |   |   |   |   |-- ✅ DOMCollection.d.ts
|   |   |   |   |   |-- ✅ DOMCollection.js
|   |   |   |   |   |-- ✅ DOMElement.d.ts
|   |   |   |   |   |-- ✅ DOMElement.js
|   |   |   |   |   |-- ✅ Immutable.d.ts
|   |   |   |   |   |-- ✅ Immutable.js
|   |   |   |   |   |-- ✅ ReactElement.d.ts
|   |   |   |   |   |-- ✅ ReactElement.js
|   |   |   |   |   |-- ✅ ReactTestComponent.d.ts
|   |   |   |   |   \-- ✅ ReactTestComponent.js
|   |   |   |   |-- ✅ collections.d.ts
|   |   |   |   |-- ✅ collections.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ types.d.ts
|   |   |   |   \-- ✅ types.js
|   |   |   |-- ✅ node_modules/
|   |   |   |   \-- ✅ ansi-styles/
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ license
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ readme.md
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ proxy-from-env/
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .travis.yml
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ test.js
|   |   |-- ✅ psl/
|   |   |   |-- ✅ data/
|   |   |   |   \-- ✅ rules.js
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ psl.cjs
|   |   |   |   |-- ✅ psl.mjs
|   |   |   |   \-- ✅ psl.umd.cjs
|   |   |   |-- ✅ types/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ test.ts
|   |   |   |   \-- ✅ tsconfig.json
|   |   |   |-- ✅ browserstack-logo.svg
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ SECURITY.md
|   |   |   \-- ✅ vite.config.js
|   |   |-- ✅ punycode/
|   |   |   |-- ✅ LICENSE-MIT.txt
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ punycode.es6.js
|   |   |   |-- ✅ punycode.js
|   |   |   \-- ✅ README.md
|   |   |-- ✅ querystringify/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ queue-microtask/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ react/
|   |   |   |-- ✅ cjs/
|   |   |   |   |-- ✅ react-jsx-dev-runtime.development.js
|   |   |   |   |-- ✅ react-jsx-dev-runtime.production.min.js
|   |   |   |   |-- ✅ react-jsx-dev-runtime.profiling.min.js
|   |   |   |   |-- ✅ react-jsx-runtime.development.js
|   |   |   |   |-- ✅ react-jsx-runtime.production.min.js
|   |   |   |   |-- ✅ react-jsx-runtime.profiling.min.js
|   |   |   |   |-- ✅ react.development.js
|   |   |   |   |-- ✅ react.production.min.js
|   |   |   |   |-- ✅ react.shared-subset.development.js
|   |   |   |   \-- ✅ react.shared-subset.production.min.js
|   |   |   |-- ✅ umd/
|   |   |   |   |-- ✅ react.development.js
|   |   |   |   |-- ✅ react.production.min.js
|   |   |   |   \-- ✅ react.profiling.min.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ jsx-dev-runtime.js
|   |   |   |-- ✅ jsx-runtime.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ react.shared-subset.js
|   |   |   \-- ✅ README.md
|   |   |-- ✅ react-chartjs-2/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ chart.d.ts
|   |   |   |   |-- ✅ chart.d.ts.map
|   |   |   |   |-- ✅ index.cjs
|   |   |   |   |-- ✅ index.cjs.map
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ index.js.map
|   |   |   |   |-- ✅ typedCharts.d.ts
|   |   |   |   |-- ✅ typedCharts.d.ts.map
|   |   |   |   |-- ✅ types.d.ts
|   |   |   |   |-- ✅ types.d.ts.map
|   |   |   |   |-- ✅ utils.d.ts
|   |   |   |   \-- ✅ utils.d.ts.map
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ react-dom/
|   |   |   |-- ✅ cjs/
|   |   |   |   |-- ✅ react-dom-server-legacy.browser.development.js
|   |   |   |   |-- ✅ react-dom-server-legacy.browser.production.min.js
|   |   |   |   |-- ✅ react-dom-server-legacy.node.development.js
|   |   |   |   |-- ✅ react-dom-server-legacy.node.production.min.js
|   |   |   |   |-- ✅ react-dom-server.browser.development.js
|   |   |   |   |-- ✅ react-dom-server.browser.production.min.js
|   |   |   |   |-- ✅ react-dom-server.node.development.js
|   |   |   |   |-- ✅ react-dom-server.node.production.min.js
|   |   |   |   |-- ✅ react-dom-test-utils.development.js
|   |   |   |   |-- ✅ react-dom-test-utils.production.min.js
|   |   |   |   |-- ✅ react-dom.development.js
|   |   |   |   |-- ✅ react-dom.production.min.js
|   |   |   |   \-- ✅ react-dom.profiling.min.js
|   |   |   |-- ✅ umd/
|   |   |   |   |-- ✅ react-dom-server-legacy.browser.development.js
|   |   |   |   |-- ✅ react-dom-server-legacy.browser.production.min.js
|   |   |   |   |-- ✅ react-dom-server.browser.development.js
|   |   |   |   |-- ✅ react-dom-server.browser.production.min.js
|   |   |   |   |-- ✅ react-dom-test-utils.development.js
|   |   |   |   |-- ✅ react-dom-test-utils.production.min.js
|   |   |   |   |-- ✅ react-dom.development.js
|   |   |   |   |-- ✅ react-dom.production.min.js
|   |   |   |   \-- ✅ react-dom.profiling.min.js
|   |   |   |-- ✅ client.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ profiling.js
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ server.browser.js
|   |   |   |-- ✅ server.js
|   |   |   |-- ✅ server.node.js
|   |   |   \-- ✅ test-utils.js
|   |   |-- ✅ react-is/
|   |   |   |-- ✅ cjs/
|   |   |   |   |-- ✅ react-is.development.js
|   |   |   |   \-- ✅ react-is.production.min.js
|   |   |   |-- ✅ umd/
|   |   |   |   |-- ✅ react-is.development.js
|   |   |   |   \-- ✅ react-is.production.min.js
|   |   |   |-- ✅ build-info.json
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ react-refresh/
|   |   |   |-- ✅ cjs/
|   |   |   |   |-- ✅ react-refresh-babel.development.js
|   |   |   |   |-- ✅ react-refresh-babel.production.js
|   |   |   |   |-- ✅ react-refresh-runtime.development.js
|   |   |   |   \-- ✅ react-refresh-runtime.production.js
|   |   |   |-- ✅ babel.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ runtime.js
|   |   |-- ✅ read-cache/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ readdirp/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ redent/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ regexp.prototype.flags/
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ builtin.js
|   |   |   |   |-- ✅ implementation.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ shimmed.js
|   |   |   |   \-- ✅ tests.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ auto.js
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ implementation.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ polyfill.js
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ shim.js
|   |   |-- ✅ requires-port/
|   |   |   |-- ✅ .npmignore
|   |   |   |-- ✅ .travis.yml
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ test.js
|   |   |-- ✅ resolve/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ bin/
|   |   |   |   \-- ✅ resolve
|   |   |   |-- ✅ example/
|   |   |   |   |-- ✅ async.js
|   |   |   |   \-- ✅ sync.js
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ async.js
|   |   |   |   |-- ✅ caller.js
|   |   |   |   |-- ✅ core.js
|   |   |   |   |-- ✅ core.json
|   |   |   |   |-- ✅ homedir.js
|   |   |   |   |-- ✅ is-core.js
|   |   |   |   |-- ✅ node-modules-paths.js
|   |   |   |   |-- ✅ normalize-options.js
|   |   |   |   \-- ✅ sync.js
|   |   |   |-- ✅ test/
|   |   |   |   |-- ✅ dotdot/
|   |   |   |   |   |-- ✅ abc/
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ module_dir/
|   |   |   |   |   |-- ✅ xmodules/
|   |   |   |   |   |   \-- ✅ aaa/
|   |   |   |   |   |       \-- ✅ index.js
|   |   |   |   |   |-- ✅ ymodules/
|   |   |   |   |   |   \-- ✅ aaa/
|   |   |   |   |   |       \-- ✅ index.js
|   |   |   |   |   \-- ✅ zmodules/
|   |   |   |   |       \-- ✅ bbb/
|   |   |   |   |           |-- ✅ main.js
|   |   |   |   |           \-- ✅ package.json
|   |   |   |   |-- ✅ node_path/
|   |   |   |   |   |-- ✅ x/
|   |   |   |   |   |   |-- ✅ aaa/
|   |   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   |   \-- ✅ ccc/
|   |   |   |   |   |       \-- ✅ index.js
|   |   |   |   |   \-- ✅ y/
|   |   |   |   |       |-- ✅ bbb/
|   |   |   |   |       |   \-- ✅ index.js
|   |   |   |   |       \-- ✅ ccc/
|   |   |   |   |           \-- ✅ index.js
|   |   |   |   |-- ✅ pathfilter/
|   |   |   |   |   \-- ✅ deep_ref/
|   |   |   |   |       \-- ✅ main.js
|   |   |   |   |-- ✅ precedence/
|   |   |   |   |   |-- ✅ aaa/
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ main.js
|   |   |   |   |   |-- ✅ bbb/
|   |   |   |   |   |   \-- ✅ main.js
|   |   |   |   |   |-- ✅ aaa.js
|   |   |   |   |   \-- ✅ bbb.js
|   |   |   |   |-- ✅ resolver/
|   |   |   |   |   |-- ✅ baz/
|   |   |   |   |   |   |-- ✅ doom.js
|   |   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |   \-- ✅ quux.js
|   |   |   |   |   |-- ✅ browser_field/
|   |   |   |   |   |   |-- ✅ a.js
|   |   |   |   |   |   |-- ✅ b.js
|   |   |   |   |   |   \-- ✅ package.json
|   |   |   |   |   |-- ✅ dot_main/
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ package.json
|   |   |   |   |   |-- ✅ dot_slash_main/
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ package.json
|   |   |   |   |   |-- ✅ false_main/
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ package.json
|   |   |   |   |   |-- ✅ incorrect_main/
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ package.json
|   |   |   |   |   |-- ✅ invalid_main/
|   |   |   |   |   |   \-- ✅ package.json
|   |   |   |   |   |-- ✅ multirepo/
|   |   |   |   |   |   |-- ✅ packages/
|   |   |   |   |   |   |   |-- ✅ package-a/
|   |   |   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |   |   \-- ✅ package.json
|   |   |   |   |   |   |   \-- ✅ package-b/
|   |   |   |   |   |   |       |-- ✅ index.js
|   |   |   |   |   |   |       \-- ✅ package.json
|   |   |   |   |   |   |-- ✅ lerna.json
|   |   |   |   |   |   \-- ✅ package.json
|   |   |   |   |   |-- ✅ nested_symlinks/
|   |   |   |   |   |   \-- ✅ mylib/
|   |   |   |   |   |       |-- ✅ async.js
|   |   |   |   |   |       |-- ✅ package.json
|   |   |   |   |   |       \-- ✅ sync.js
|   |   |   |   |   |-- ✅ other_path/
|   |   |   |   |   |   |-- ✅ lib/
|   |   |   |   |   |   |   \-- ✅ other-lib.js
|   |   |   |   |   |   \-- ✅ root.js
|   |   |   |   |   |-- ✅ quux/
|   |   |   |   |   |   \-- ✅ foo/
|   |   |   |   |   |       \-- ✅ index.js
|   |   |   |   |   |-- ✅ same_names/
|   |   |   |   |   |   |-- ✅ foo/
|   |   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   |   \-- ✅ foo.js
|   |   |   |   |   |-- ✅ symlinked/
|   |   |   |   |   |   |-- ✅ _/
|   |   |   |   |   |   |   |-- ✅ node_modules/
|   |   |   |   |   |   |   |   \-- ✅ foo.js
|   |   |   |   |   |   |   \-- ✅ symlink_target/
|   |   |   |   |   |   |       \-- ✅ .gitkeep
|   |   |   |   |   |   \-- ✅ package/
|   |   |   |   |   |       |-- ✅ bar.js
|   |   |   |   |   |       \-- ✅ package.json
|   |   |   |   |   |-- ✅ without_basedir/
|   |   |   |   |   |   \-- ✅ main.js
|   |   |   |   |   |-- ✅ cup.coffee
|   |   |   |   |   |-- ✅ foo.js
|   |   |   |   |   |-- ✅ mug.coffee
|   |   |   |   |   \-- ✅ mug.js
|   |   |   |   |-- ✅ shadowed_core/
|   |   |   |   |   \-- ✅ node_modules/
|   |   |   |   |       \-- ✅ util/
|   |   |   |   |           \-- ✅ index.js
|   |   |   |   |-- ✅ core.js
|   |   |   |   |-- ✅ dotdot.js
|   |   |   |   |-- ✅ faulty_basedir.js
|   |   |   |   |-- ✅ filter.js
|   |   |   |   |-- ✅ filter_sync.js
|   |   |   |   |-- ✅ home_paths.js
|   |   |   |   |-- ✅ home_paths_sync.js
|   |   |   |   |-- ✅ mock.js
|   |   |   |   |-- ✅ mock_sync.js
|   |   |   |   |-- ✅ module_dir.js
|   |   |   |   |-- ✅ node-modules-paths.js
|   |   |   |   |-- ✅ node_path.js
|   |   |   |   |-- ✅ nonstring.js
|   |   |   |   |-- ✅ pathfilter.js
|   |   |   |   |-- ✅ precedence.js
|   |   |   |   |-- ✅ resolver.js
|   |   |   |   |-- ✅ resolver_sync.js
|   |   |   |   |-- ✅ shadowed_core.js
|   |   |   |   |-- ✅ subdirs.js
|   |   |   |   \-- ✅ symlinks.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ async.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ readme.markdown
|   |   |   |-- ✅ SECURITY.md
|   |   |   \-- ✅ sync.js
|   |   |-- ✅ reusify/
|   |   |   |-- ✅ .github/
|   |   |   |   |-- ✅ workflows/
|   |   |   |   |   \-- ✅ ci.yml
|   |   |   |   \-- ✅ dependabot.yml
|   |   |   |-- ✅ benchmarks/
|   |   |   |   |-- ✅ createNoCodeFunction.js
|   |   |   |   |-- ✅ fib.js
|   |   |   |   \-- ✅ reuseNoCodeFunction.js
|   |   |   |-- ✅ eslint.config.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ reusify.d.ts
|   |   |   |-- ✅ reusify.js
|   |   |   |-- ✅ SECURITY.md
|   |   |   |-- ✅ test.js
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ rollup/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ bin/
|   |   |   |   |   \-- ✅ rollup
|   |   |   |   |-- ✅ es/
|   |   |   |   |   |-- ✅ shared/
|   |   |   |   |   |   |-- ✅ node-entry.js
|   |   |   |   |   |   |-- ✅ parseAst.js
|   |   |   |   |   |   \-- ✅ watch.js
|   |   |   |   |   |-- ✅ getLogFilter.js
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |-- ✅ parseAst.js
|   |   |   |   |   \-- ✅ rollup.js
|   |   |   |   |-- ✅ shared/
|   |   |   |   |   |-- ✅ fsevents-importer.js
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ loadConfigFile.js
|   |   |   |   |   |-- ✅ parseAst.js
|   |   |   |   |   |-- ✅ rollup.js
|   |   |   |   |   |-- ✅ watch-cli.js
|   |   |   |   |   \-- ✅ watch.js
|   |   |   |   |-- ✅ getLogFilter.d.ts
|   |   |   |   |-- ✅ getLogFilter.js
|   |   |   |   |-- ✅ loadConfigFile.d.ts
|   |   |   |   |-- ✅ loadConfigFile.js
|   |   |   |   |-- ✅ native.js
|   |   |   |   |-- ✅ parseAst.d.ts
|   |   |   |   |-- ✅ parseAst.js
|   |   |   |   |-- ✅ rollup.d.ts
|   |   |   |   \-- ✅ rollup.js
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ rrweb-cssom/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ clone.js
|   |   |   |   |-- ✅ CSSConditionRule.js
|   |   |   |   |-- ✅ CSSDocumentRule.js
|   |   |   |   |-- ✅ CSSFontFaceRule.js
|   |   |   |   |-- ✅ CSSGroupingRule.js
|   |   |   |   |-- ✅ CSSHostRule.js
|   |   |   |   |-- ✅ CSSImportRule.js
|   |   |   |   |-- ✅ CSSKeyframeRule.js
|   |   |   |   |-- ✅ CSSKeyframesRule.js
|   |   |   |   |-- ✅ CSSMediaRule.js
|   |   |   |   |-- ✅ CSSOM.js
|   |   |   |   |-- ✅ CSSRule.js
|   |   |   |   |-- ✅ CSSStyleDeclaration.js
|   |   |   |   |-- ✅ CSSStyleRule.js
|   |   |   |   |-- ✅ CSSStyleSheet.js
|   |   |   |   |-- ✅ CSSSupportsRule.js
|   |   |   |   |-- ✅ CSSValue.js
|   |   |   |   |-- ✅ CSSValueExpression.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ MatcherList.js
|   |   |   |   |-- ✅ MediaList.js
|   |   |   |   |-- ✅ parse.js
|   |   |   |   \-- ✅ StyleSheet.js
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.mdown
|   |   |-- ✅ run-parallel/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ safe-regex-test/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ safer-buffer/
|   |   |   |-- ✅ dangerous.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ Porting-Buffer.md
|   |   |   |-- ✅ Readme.md
|   |   |   |-- ✅ safer.js
|   |   |   \-- ✅ tests.js
|   |   |-- ✅ saxes/
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ saxes.d.ts
|   |   |   |-- ✅ saxes.js
|   |   |   \-- ✅ saxes.js.map
|   |   |-- ✅ scheduler/
|   |   |   |-- ✅ cjs/
|   |   |   |   |-- ✅ scheduler-unstable_mock.development.js
|   |   |   |   |-- ✅ scheduler-unstable_mock.production.min.js
|   |   |   |   |-- ✅ scheduler-unstable_post_task.development.js
|   |   |   |   |-- ✅ scheduler-unstable_post_task.production.min.js
|   |   |   |   |-- ✅ scheduler.development.js
|   |   |   |   \-- ✅ scheduler.production.min.js
|   |   |   |-- ✅ umd/
|   |   |   |   |-- ✅ scheduler-unstable_mock.development.js
|   |   |   |   |-- ✅ scheduler-unstable_mock.production.min.js
|   |   |   |   |-- ✅ scheduler.development.js
|   |   |   |   |-- ✅ scheduler.production.min.js
|   |   |   |   \-- ✅ scheduler.profiling.min.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ unstable_mock.js
|   |   |   \-- ✅ unstable_post_task.js
|   |   |-- ✅ semver/
|   |   |   |-- ✅ bin/
|   |   |   |   \-- ✅ semver.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ range.bnf
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ semver.js
|   |   |-- ✅ set-function-length/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ env.d.ts
|   |   |   |-- ✅ env.js
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ set-function-name/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ shebang-command/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ shebang-regex/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ side-channel/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ side-channel-list/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ list.d.ts
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ side-channel-map/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ side-channel-weakmap/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ siginfo/
|   |   |   |-- ✅ .travis.yml
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ test.js
|   |   |-- ✅ signal-exit/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ cjs/
|   |   |   |   |   |-- ✅ browser.d.ts
|   |   |   |   |   |-- ✅ browser.d.ts.map
|   |   |   |   |   |-- ✅ browser.js
|   |   |   |   |   |-- ✅ browser.js.map
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.d.ts.map
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ index.js.map
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |-- ✅ signals.d.ts
|   |   |   |   |   |-- ✅ signals.d.ts.map
|   |   |   |   |   |-- ✅ signals.js
|   |   |   |   |   \-- ✅ signals.js.map
|   |   |   |   \-- ✅ mjs/
|   |   |   |       |-- ✅ browser.d.ts
|   |   |   |       |-- ✅ browser.d.ts.map
|   |   |   |       |-- ✅ browser.js
|   |   |   |       |-- ✅ browser.js.map
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.d.ts.map
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ index.js.map
|   |   |   |       |-- ✅ package.json
|   |   |   |       |-- ✅ signals.d.ts
|   |   |   |       |-- ✅ signals.d.ts.map
|   |   |   |       |-- ✅ signals.js
|   |   |   |       \-- ✅ signals.js.map
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ slash/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ source-map-js/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ array-set.js
|   |   |   |   |-- ✅ base64-vlq.js
|   |   |   |   |-- ✅ base64.js
|   |   |   |   |-- ✅ binary-search.js
|   |   |   |   |-- ✅ mapping-list.js
|   |   |   |   |-- ✅ quick-sort.js
|   |   |   |   |-- ✅ source-map-consumer.d.ts
|   |   |   |   |-- ✅ source-map-consumer.js
|   |   |   |   |-- ✅ source-map-generator.d.ts
|   |   |   |   |-- ✅ source-map-generator.js
|   |   |   |   |-- ✅ source-node.d.ts
|   |   |   |   |-- ✅ source-node.js
|   |   |   |   \-- ✅ util.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ source-map.d.ts
|   |   |   \-- ✅ source-map.js
|   |   |-- ✅ stack-utils/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ stackback/
|   |   |   |-- ✅ .npmignore
|   |   |   |-- ✅ .travis.yml
|   |   |   |-- ✅ formatstack.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ test.js
|   |   |-- ✅ std-env/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ index.cjs
|   |   |   |   |-- ✅ index.d.cts
|   |   |   |   |-- ✅ index.d.mts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   \-- ✅ index.mjs
|   |   |   |-- ✅ LICENCE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ stop-iteration-iterator/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ string-width/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ string-width-cjs/
|   |   |   |-- ✅ node_modules/
|   |   |   |   |-- ✅ emoji-regex/
|   |   |   |   |   |-- ✅ es2015/
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ text.js
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ LICENSE-MIT.txt
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |-- ✅ README.md
|   |   |   |   |   \-- ✅ text.js
|   |   |   |   \-- ✅ strip-ansi/
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ license
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ readme.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ strip-ansi/
|   |   |   |-- ✅ node_modules/
|   |   |   |   \-- ✅ ansi-regex/
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ license
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ readme.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ strip-ansi-cjs/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ strip-indent/
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ strip-literal/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ index.cjs
|   |   |   |   |-- ✅ index.d.cts
|   |   |   |   |-- ✅ index.d.mts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   \-- ✅ index.mjs
|   |   |   |-- ✅ node_modules/
|   |   |   |   \-- ✅ js-tokens/
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ LICENSE
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ README.md
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ sucrase/
|   |   |   |-- ✅ bin/
|   |   |   |   |-- ✅ sucrase
|   |   |   |   \-- ✅ sucrase-node
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ esm/
|   |   |   |   |   |-- ✅ parser/
|   |   |   |   |   |   |-- ✅ plugins/
|   |   |   |   |   |   |   |-- ✅ jsx/
|   |   |   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |   |   \-- ✅ xhtml.js
|   |   |   |   |   |   |   |-- ✅ flow.js
|   |   |   |   |   |   |   |-- ✅ types.js
|   |   |   |   |   |   |   \-- ✅ typescript.js
|   |   |   |   |   |   |-- ✅ tokenizer/
|   |   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |   |-- ✅ keywords.js
|   |   |   |   |   |   |   |-- ✅ readWord.js
|   |   |   |   |   |   |   |-- ✅ readWordTree.js
|   |   |   |   |   |   |   |-- ✅ state.js
|   |   |   |   |   |   |   \-- ✅ types.js
|   |   |   |   |   |   |-- ✅ traverser/
|   |   |   |   |   |   |   |-- ✅ base.js
|   |   |   |   |   |   |   |-- ✅ expression.js
|   |   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |   |-- ✅ lval.js
|   |   |   |   |   |   |   |-- ✅ statement.js
|   |   |   |   |   |   |   \-- ✅ util.js
|   |   |   |   |   |   |-- ✅ util/
|   |   |   |   |   |   |   |-- ✅ charcodes.js
|   |   |   |   |   |   |   |-- ✅ identifier.js
|   |   |   |   |   |   |   \-- ✅ whitespace.js
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   |-- ✅ transformers/
|   |   |   |   |   |   |-- ✅ CJSImportTransformer.js
|   |   |   |   |   |   |-- ✅ ESMImportTransformer.js
|   |   |   |   |   |   |-- ✅ FlowTransformer.js
|   |   |   |   |   |   |-- ✅ JestHoistTransformer.js
|   |   |   |   |   |   |-- ✅ JSXTransformer.js
|   |   |   |   |   |   |-- ✅ NumericSeparatorTransformer.js
|   |   |   |   |   |   |-- ✅ OptionalCatchBindingTransformer.js
|   |   |   |   |   |   |-- ✅ OptionalChainingNullishTransformer.js
|   |   |   |   |   |   |-- ✅ ReactDisplayNameTransformer.js
|   |   |   |   |   |   |-- ✅ ReactHotLoaderTransformer.js
|   |   |   |   |   |   |-- ✅ RootTransformer.js
|   |   |   |   |   |   |-- ✅ Transformer.js
|   |   |   |   |   |   \-- ✅ TypeScriptTransformer.js
|   |   |   |   |   |-- ✅ util/
|   |   |   |   |   |   |-- ✅ elideImportEquals.js
|   |   |   |   |   |   |-- ✅ formatTokens.js
|   |   |   |   |   |   |-- ✅ getClassInfo.js
|   |   |   |   |   |   |-- ✅ getDeclarationInfo.js
|   |   |   |   |   |   |-- ✅ getIdentifierNames.js
|   |   |   |   |   |   |-- ✅ getImportExportSpecifierInfo.js
|   |   |   |   |   |   |-- ✅ getJSXPragmaInfo.js
|   |   |   |   |   |   |-- ✅ getNonTypeIdentifiers.js
|   |   |   |   |   |   |-- ✅ getTSImportedNames.js
|   |   |   |   |   |   |-- ✅ isAsyncOperation.js
|   |   |   |   |   |   |-- ✅ isExportFrom.js
|   |   |   |   |   |   |-- ✅ isIdentifier.js
|   |   |   |   |   |   |-- ✅ removeMaybeImportAttributes.js
|   |   |   |   |   |   \-- ✅ shouldElideDefaultExport.js
|   |   |   |   |   |-- ✅ CJSImportProcessor.js
|   |   |   |   |   |-- ✅ cli.js
|   |   |   |   |   |-- ✅ computeSourceMap.js
|   |   |   |   |   |-- ✅ HelperManager.js
|   |   |   |   |   |-- ✅ identifyShadowedGlobals.js
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ NameManager.js
|   |   |   |   |   |-- ✅ Options-gen-types.js
|   |   |   |   |   |-- ✅ Options.js
|   |   |   |   |   |-- ✅ register.js
|   |   |   |   |   \-- ✅ TokenProcessor.js
|   |   |   |   |-- ✅ parser/
|   |   |   |   |   |-- ✅ plugins/
|   |   |   |   |   |   |-- ✅ jsx/
|   |   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |   \-- ✅ xhtml.js
|   |   |   |   |   |   |-- ✅ flow.js
|   |   |   |   |   |   |-- ✅ types.js
|   |   |   |   |   |   \-- ✅ typescript.js
|   |   |   |   |   |-- ✅ tokenizer/
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ keywords.js
|   |   |   |   |   |   |-- ✅ readWord.js
|   |   |   |   |   |   |-- ✅ readWordTree.js
|   |   |   |   |   |   |-- ✅ state.js
|   |   |   |   |   |   \-- ✅ types.js
|   |   |   |   |   |-- ✅ traverser/
|   |   |   |   |   |   |-- ✅ base.js
|   |   |   |   |   |   |-- ✅ expression.js
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ lval.js
|   |   |   |   |   |   |-- ✅ statement.js
|   |   |   |   |   |   \-- ✅ util.js
|   |   |   |   |   |-- ✅ util/
|   |   |   |   |   |   |-- ✅ charcodes.js
|   |   |   |   |   |   |-- ✅ identifier.js
|   |   |   |   |   |   \-- ✅ whitespace.js
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ transformers/
|   |   |   |   |   |-- ✅ CJSImportTransformer.js
|   |   |   |   |   |-- ✅ ESMImportTransformer.js
|   |   |   |   |   |-- ✅ FlowTransformer.js
|   |   |   |   |   |-- ✅ JestHoistTransformer.js
|   |   |   |   |   |-- ✅ JSXTransformer.js
|   |   |   |   |   |-- ✅ NumericSeparatorTransformer.js
|   |   |   |   |   |-- ✅ OptionalCatchBindingTransformer.js
|   |   |   |   |   |-- ✅ OptionalChainingNullishTransformer.js
|   |   |   |   |   |-- ✅ ReactDisplayNameTransformer.js
|   |   |   |   |   |-- ✅ ReactHotLoaderTransformer.js
|   |   |   |   |   |-- ✅ RootTransformer.js
|   |   |   |   |   |-- ✅ Transformer.js
|   |   |   |   |   \-- ✅ TypeScriptTransformer.js
|   |   |   |   |-- ✅ types/
|   |   |   |   |   |-- ✅ parser/
|   |   |   |   |   |   |-- ✅ plugins/
|   |   |   |   |   |   |   |-- ✅ jsx/
|   |   |   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   |   |   \-- ✅ xhtml.d.ts
|   |   |   |   |   |   |   |-- ✅ flow.d.ts
|   |   |   |   |   |   |   |-- ✅ types.d.ts
|   |   |   |   |   |   |   \-- ✅ typescript.d.ts
|   |   |   |   |   |   |-- ✅ tokenizer/
|   |   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   |   |-- ✅ keywords.d.ts
|   |   |   |   |   |   |   |-- ✅ readWord.d.ts
|   |   |   |   |   |   |   |-- ✅ readWordTree.d.ts
|   |   |   |   |   |   |   |-- ✅ state.d.ts
|   |   |   |   |   |   |   \-- ✅ types.d.ts
|   |   |   |   |   |   |-- ✅ traverser/
|   |   |   |   |   |   |   |-- ✅ base.d.ts
|   |   |   |   |   |   |   |-- ✅ expression.d.ts
|   |   |   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |   |   |-- ✅ lval.d.ts
|   |   |   |   |   |   |   |-- ✅ statement.d.ts
|   |   |   |   |   |   |   \-- ✅ util.d.ts
|   |   |   |   |   |   |-- ✅ util/
|   |   |   |   |   |   |   |-- ✅ charcodes.d.ts
|   |   |   |   |   |   |   |-- ✅ identifier.d.ts
|   |   |   |   |   |   |   \-- ✅ whitespace.d.ts
|   |   |   |   |   |   \-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ transformers/
|   |   |   |   |   |   |-- ✅ CJSImportTransformer.d.ts
|   |   |   |   |   |   |-- ✅ ESMImportTransformer.d.ts
|   |   |   |   |   |   |-- ✅ FlowTransformer.d.ts
|   |   |   |   |   |   |-- ✅ JestHoistTransformer.d.ts
|   |   |   |   |   |   |-- ✅ JSXTransformer.d.ts
|   |   |   |   |   |   |-- ✅ NumericSeparatorTransformer.d.ts
|   |   |   |   |   |   |-- ✅ OptionalCatchBindingTransformer.d.ts
|   |   |   |   |   |   |-- ✅ OptionalChainingNullishTransformer.d.ts
|   |   |   |   |   |   |-- ✅ ReactDisplayNameTransformer.d.ts
|   |   |   |   |   |   |-- ✅ ReactHotLoaderTransformer.d.ts
|   |   |   |   |   |   |-- ✅ RootTransformer.d.ts
|   |   |   |   |   |   |-- ✅ Transformer.d.ts
|   |   |   |   |   |   \-- ✅ TypeScriptTransformer.d.ts
|   |   |   |   |   |-- ✅ util/
|   |   |   |   |   |   |-- ✅ elideImportEquals.d.ts
|   |   |   |   |   |   |-- ✅ formatTokens.d.ts
|   |   |   |   |   |   |-- ✅ getClassInfo.d.ts
|   |   |   |   |   |   |-- ✅ getDeclarationInfo.d.ts
|   |   |   |   |   |   |-- ✅ getIdentifierNames.d.ts
|   |   |   |   |   |   |-- ✅ getImportExportSpecifierInfo.d.ts
|   |   |   |   |   |   |-- ✅ getJSXPragmaInfo.d.ts
|   |   |   |   |   |   |-- ✅ getNonTypeIdentifiers.d.ts
|   |   |   |   |   |   |-- ✅ getTSImportedNames.d.ts
|   |   |   |   |   |   |-- ✅ isAsyncOperation.d.ts
|   |   |   |   |   |   |-- ✅ isExportFrom.d.ts
|   |   |   |   |   |   |-- ✅ isIdentifier.d.ts
|   |   |   |   |   |   |-- ✅ removeMaybeImportAttributes.d.ts
|   |   |   |   |   |   \-- ✅ shouldElideDefaultExport.d.ts
|   |   |   |   |   |-- ✅ CJSImportProcessor.d.ts
|   |   |   |   |   |-- ✅ cli.d.ts
|   |   |   |   |   |-- ✅ computeSourceMap.d.ts
|   |   |   |   |   |-- ✅ HelperManager.d.ts
|   |   |   |   |   |-- ✅ identifyShadowedGlobals.d.ts
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ NameManager.d.ts
|   |   |   |   |   |-- ✅ Options-gen-types.d.ts
|   |   |   |   |   |-- ✅ Options.d.ts
|   |   |   |   |   |-- ✅ register.d.ts
|   |   |   |   |   \-- ✅ TokenProcessor.d.ts
|   |   |   |   |-- ✅ util/
|   |   |   |   |   |-- ✅ elideImportEquals.js
|   |   |   |   |   |-- ✅ formatTokens.js
|   |   |   |   |   |-- ✅ getClassInfo.js
|   |   |   |   |   |-- ✅ getDeclarationInfo.js
|   |   |   |   |   |-- ✅ getIdentifierNames.js
|   |   |   |   |   |-- ✅ getImportExportSpecifierInfo.js
|   |   |   |   |   |-- ✅ getJSXPragmaInfo.js
|   |   |   |   |   |-- ✅ getNonTypeIdentifiers.js
|   |   |   |   |   |-- ✅ getTSImportedNames.js
|   |   |   |   |   |-- ✅ isAsyncOperation.js
|   |   |   |   |   |-- ✅ isExportFrom.js
|   |   |   |   |   |-- ✅ isIdentifier.js
|   |   |   |   |   |-- ✅ removeMaybeImportAttributes.js
|   |   |   |   |   \-- ✅ shouldElideDefaultExport.js
|   |   |   |   |-- ✅ CJSImportProcessor.js
|   |   |   |   |-- ✅ cli.js
|   |   |   |   |-- ✅ computeSourceMap.js
|   |   |   |   |-- ✅ HelperManager.js
|   |   |   |   |-- ✅ identifyShadowedGlobals.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ NameManager.js
|   |   |   |   |-- ✅ Options-gen-types.js
|   |   |   |   |-- ✅ Options.js
|   |   |   |   |-- ✅ register.js
|   |   |   |   \-- ✅ TokenProcessor.js
|   |   |   |-- ✅ register/
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ js.js
|   |   |   |   |-- ✅ jsx.js
|   |   |   |   |-- ✅ ts-legacy-module-interop.js
|   |   |   |   |-- ✅ ts.js
|   |   |   |   |-- ✅ tsx-legacy-module-interop.js
|   |   |   |   \-- ✅ tsx.js
|   |   |   |-- ✅ ts-node-plugin/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ supports-color/
|   |   |   |-- ✅ browser.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ supports-preserve-symlinks-flag/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ browser.js
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ symbol-tree/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ SymbolTree.js
|   |   |   |   |-- ✅ SymbolTreeNode.js
|   |   |   |   |-- ✅ TreeIterator.js
|   |   |   |   \-- ✅ TreePosition.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ tailwindcss/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ cli/
|   |   |   |   |   |-- ✅ build/
|   |   |   |   |   |   |-- ✅ deps.js
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ plugin.js
|   |   |   |   |   |   |-- ✅ utils.js
|   |   |   |   |   |   \-- ✅ watching.js
|   |   |   |   |   |-- ✅ help/
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   |-- ✅ init/
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ css/
|   |   |   |   |   |-- ✅ LICENSE
|   |   |   |   |   \-- ✅ preflight.css
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ cacheInvalidation.js
|   |   |   |   |   |-- ✅ collapseAdjacentRules.js
|   |   |   |   |   |-- ✅ collapseDuplicateDeclarations.js
|   |   |   |   |   |-- ✅ content.js
|   |   |   |   |   |-- ✅ defaultExtractor.js
|   |   |   |   |   |-- ✅ evaluateTailwindFunctions.js
|   |   |   |   |   |-- ✅ expandApplyAtRules.js
|   |   |   |   |   |-- ✅ expandTailwindAtRules.js
|   |   |   |   |   |-- ✅ findAtConfigPath.js
|   |   |   |   |   |-- ✅ generateRules.js
|   |   |   |   |   |-- ✅ getModuleDependencies.js
|   |   |   |   |   |-- ✅ load-config.js
|   |   |   |   |   |-- ✅ normalizeTailwindDirectives.js
|   |   |   |   |   |-- ✅ offsets.js
|   |   |   |   |   |-- ✅ partitionApplyAtRules.js
|   |   |   |   |   |-- ✅ regex.js
|   |   |   |   |   |-- ✅ remap-bitfield.js
|   |   |   |   |   |-- ✅ resolveDefaultsAtRules.js
|   |   |   |   |   |-- ✅ setupContextUtils.js
|   |   |   |   |   |-- ✅ setupTrackingContext.js
|   |   |   |   |   |-- ✅ sharedState.js
|   |   |   |   |   \-- ✅ substituteScreenAtRules.js
|   |   |   |   |-- ✅ postcss-plugins/
|   |   |   |   |   \-- ✅ nesting/
|   |   |   |   |       |-- ✅ index.js
|   |   |   |   |       |-- ✅ plugin.js
|   |   |   |   |       \-- ✅ README.md
|   |   |   |   |-- ✅ public/
|   |   |   |   |   |-- ✅ colors.js
|   |   |   |   |   |-- ✅ create-plugin.js
|   |   |   |   |   |-- ✅ default-config.js
|   |   |   |   |   |-- ✅ default-theme.js
|   |   |   |   |   |-- ✅ load-config.js
|   |   |   |   |   \-- ✅ resolve-config.js
|   |   |   |   |-- ✅ util/
|   |   |   |   |   |-- ✅ applyImportantSelector.js
|   |   |   |   |   |-- ✅ bigSign.js
|   |   |   |   |   |-- ✅ buildMediaQuery.js
|   |   |   |   |   |-- ✅ cloneDeep.js
|   |   |   |   |   |-- ✅ cloneNodes.js
|   |   |   |   |   |-- ✅ color.js
|   |   |   |   |   |-- ✅ colorNames.js
|   |   |   |   |   |-- ✅ configurePlugins.js
|   |   |   |   |   |-- ✅ createPlugin.js
|   |   |   |   |   |-- ✅ createUtilityPlugin.js
|   |   |   |   |   |-- ✅ dataTypes.js
|   |   |   |   |   |-- ✅ defaults.js
|   |   |   |   |   |-- ✅ escapeClassName.js
|   |   |   |   |   |-- ✅ escapeCommas.js
|   |   |   |   |   |-- ✅ flattenColorPalette.js
|   |   |   |   |   |-- ✅ formatVariantSelector.js
|   |   |   |   |   |-- ✅ getAllConfigs.js
|   |   |   |   |   |-- ✅ hashConfig.js
|   |   |   |   |   |-- ✅ isKeyframeRule.js
|   |   |   |   |   |-- ✅ isPlainObject.js
|   |   |   |   |   |-- ✅ isSyntacticallyValidPropertyValue.js
|   |   |   |   |   |-- ✅ log.js
|   |   |   |   |   |-- ✅ nameClass.js
|   |   |   |   |   |-- ✅ negateValue.js
|   |   |   |   |   |-- ✅ normalizeConfig.js
|   |   |   |   |   |-- ✅ normalizeScreens.js
|   |   |   |   |   |-- ✅ parseAnimationValue.js
|   |   |   |   |   |-- ✅ parseBoxShadowValue.js
|   |   |   |   |   |-- ✅ parseDependency.js
|   |   |   |   |   |-- ✅ parseGlob.js
|   |   |   |   |   |-- ✅ parseObjectStyles.js
|   |   |   |   |   |-- ✅ pluginUtils.js
|   |   |   |   |   |-- ✅ prefixSelector.js
|   |   |   |   |   |-- ✅ pseudoElements.js
|   |   |   |   |   |-- ✅ removeAlphaVariables.js
|   |   |   |   |   |-- ✅ resolveConfig.js
|   |   |   |   |   |-- ✅ resolveConfigPath.js
|   |   |   |   |   |-- ✅ responsive.js
|   |   |   |   |   |-- ✅ splitAtTopLevelOnly.js
|   |   |   |   |   |-- ✅ tap.js
|   |   |   |   |   |-- ✅ toColorValue.js
|   |   |   |   |   |-- ✅ toPath.js
|   |   |   |   |   |-- ✅ transformThemeValue.js
|   |   |   |   |   |-- ✅ validateConfig.js
|   |   |   |   |   |-- ✅ validateFormalSyntax.js
|   |   |   |   |   \-- ✅ withAlphaVariable.js
|   |   |   |   |-- ✅ value-parser/
|   |   |   |   |   |-- ✅ index.d.js
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ LICENSE
|   |   |   |   |   |-- ✅ parse.js
|   |   |   |   |   |-- ✅ README.md
|   |   |   |   |   |-- ✅ stringify.js
|   |   |   |   |   |-- ✅ unit.js
|   |   |   |   |   \-- ✅ walk.js
|   |   |   |   |-- ✅ cli-peer-dependencies.js
|   |   |   |   |-- ✅ cli.js
|   |   |   |   |-- ✅ corePluginList.js
|   |   |   |   |-- ✅ corePlugins.js
|   |   |   |   |-- ✅ featureFlags.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ plugin.js
|   |   |   |   \-- ✅ processTailwindFeatures.js
|   |   |   |-- ✅ nesting/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ peers/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ scripts/
|   |   |   |   |-- ✅ create-plugin-list.js
|   |   |   |   |-- ✅ generate-types.js
|   |   |   |   |-- ✅ release-channel.js
|   |   |   |   |-- ✅ release-notes.js
|   |   |   |   \-- ✅ type-utils.js
|   |   |   |-- ✅ src/
|   |   |   |   |-- ✅ cli/
|   |   |   |   |   |-- ✅ build/
|   |   |   |   |   |   |-- ✅ deps.js
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   |-- ✅ plugin.js
|   |   |   |   |   |   |-- ✅ utils.js
|   |   |   |   |   |   \-- ✅ watching.js
|   |   |   |   |   |-- ✅ help/
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   |-- ✅ init/
|   |   |   |   |   |   \-- ✅ index.js
|   |   |   |   |   \-- ✅ index.js
|   |   |   |   |-- ✅ css/
|   |   |   |   |   |-- ✅ LICENSE
|   |   |   |   |   \-- ✅ preflight.css
|   |   |   |   |-- ✅ lib/
|   |   |   |   |   |-- ✅ cacheInvalidation.js
|   |   |   |   |   |-- ✅ collapseAdjacentRules.js
|   |   |   |   |   |-- ✅ collapseDuplicateDeclarations.js
|   |   |   |   |   |-- ✅ content.js
|   |   |   |   |   |-- ✅ defaultExtractor.js
|   |   |   |   |   |-- ✅ evaluateTailwindFunctions.js
|   |   |   |   |   |-- ✅ expandApplyAtRules.js
|   |   |   |   |   |-- ✅ expandTailwindAtRules.js
|   |   |   |   |   |-- ✅ findAtConfigPath.js
|   |   |   |   |   |-- ✅ generateRules.js
|   |   |   |   |   |-- ✅ getModuleDependencies.js
|   |   |   |   |   |-- ✅ load-config.ts
|   |   |   |   |   |-- ✅ normalizeTailwindDirectives.js
|   |   |   |   |   |-- ✅ offsets.js
|   |   |   |   |   |-- ✅ partitionApplyAtRules.js
|   |   |   |   |   |-- ✅ regex.js
|   |   |   |   |   |-- ✅ remap-bitfield.js
|   |   |   |   |   |-- ✅ resolveDefaultsAtRules.js
|   |   |   |   |   |-- ✅ setupContextUtils.js
|   |   |   |   |   |-- ✅ setupTrackingContext.js
|   |   |   |   |   |-- ✅ sharedState.js
|   |   |   |   |   \-- ✅ substituteScreenAtRules.js
|   |   |   |   |-- ✅ postcss-plugins/
|   |   |   |   |   \-- ✅ nesting/
|   |   |   |   |       |-- ✅ index.js
|   |   |   |   |       |-- ✅ plugin.js
|   |   |   |   |       \-- ✅ README.md
|   |   |   |   |-- ✅ public/
|   |   |   |   |   |-- ✅ colors.js
|   |   |   |   |   |-- ✅ create-plugin.js
|   |   |   |   |   |-- ✅ default-config.js
|   |   |   |   |   |-- ✅ default-theme.js
|   |   |   |   |   |-- ✅ load-config.js
|   |   |   |   |   \-- ✅ resolve-config.js
|   |   |   |   |-- ✅ util/
|   |   |   |   |   |-- ✅ applyImportantSelector.js
|   |   |   |   |   |-- ✅ bigSign.js
|   |   |   |   |   |-- ✅ buildMediaQuery.js
|   |   |   |   |   |-- ✅ cloneDeep.js
|   |   |   |   |   |-- ✅ cloneNodes.js
|   |   |   |   |   |-- ✅ color.js
|   |   |   |   |   |-- ✅ colorNames.js
|   |   |   |   |   |-- ✅ configurePlugins.js
|   |   |   |   |   |-- ✅ createPlugin.js
|   |   |   |   |   |-- ✅ createUtilityPlugin.js
|   |   |   |   |   |-- ✅ dataTypes.js
|   |   |   |   |   |-- ✅ defaults.js
|   |   |   |   |   |-- ✅ escapeClassName.js
|   |   |   |   |   |-- ✅ escapeCommas.js
|   |   |   |   |   |-- ✅ flattenColorPalette.js
|   |   |   |   |   |-- ✅ formatVariantSelector.js
|   |   |   |   |   |-- ✅ getAllConfigs.js
|   |   |   |   |   |-- ✅ hashConfig.js
|   |   |   |   |   |-- ✅ isKeyframeRule.js
|   |   |   |   |   |-- ✅ isPlainObject.js
|   |   |   |   |   |-- ✅ isSyntacticallyValidPropertyValue.js
|   |   |   |   |   |-- ✅ log.js
|   |   |   |   |   |-- ✅ nameClass.js
|   |   |   |   |   |-- ✅ negateValue.js
|   |   |   |   |   |-- ✅ normalizeConfig.js
|   |   |   |   |   |-- ✅ normalizeScreens.js
|   |   |   |   |   |-- ✅ parseAnimationValue.js
|   |   |   |   |   |-- ✅ parseBoxShadowValue.js
|   |   |   |   |   |-- ✅ parseDependency.js
|   |   |   |   |   |-- ✅ parseGlob.js
|   |   |   |   |   |-- ✅ parseObjectStyles.js
|   |   |   |   |   |-- ✅ pluginUtils.js
|   |   |   |   |   |-- ✅ prefixSelector.js
|   |   |   |   |   |-- ✅ pseudoElements.js
|   |   |   |   |   |-- ✅ removeAlphaVariables.js
|   |   |   |   |   |-- ✅ resolveConfig.js
|   |   |   |   |   |-- ✅ resolveConfigPath.js
|   |   |   |   |   |-- ✅ responsive.js
|   |   |   |   |   |-- ✅ splitAtTopLevelOnly.js
|   |   |   |   |   |-- ✅ tap.js
|   |   |   |   |   |-- ✅ toColorValue.js
|   |   |   |   |   |-- ✅ toPath.js
|   |   |   |   |   |-- ✅ transformThemeValue.js
|   |   |   |   |   |-- ✅ validateConfig.js
|   |   |   |   |   |-- ✅ validateFormalSyntax.js
|   |   |   |   |   \-- ✅ withAlphaVariable.js
|   |   |   |   |-- ✅ value-parser/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ LICENSE
|   |   |   |   |   |-- ✅ parse.js
|   |   |   |   |   |-- ✅ README.md
|   |   |   |   |   |-- ✅ stringify.js
|   |   |   |   |   |-- ✅ unit.js
|   |   |   |   |   \-- ✅ walk.js
|   |   |   |   |-- ✅ cli-peer-dependencies.js
|   |   |   |   |-- ✅ cli.js
|   |   |   |   |-- ✅ corePluginList.js
|   |   |   |   |-- ✅ corePlugins.js
|   |   |   |   |-- ✅ featureFlags.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ plugin.js
|   |   |   |   \-- ✅ processTailwindFeatures.js
|   |   |   |-- ✅ stubs/
|   |   |   |   |-- ✅ .npmignore
|   |   |   |   |-- ✅ .prettierrc.json
|   |   |   |   |-- ✅ config.full.js
|   |   |   |   |-- ✅ config.simple.js
|   |   |   |   |-- ✅ postcss.config.cjs
|   |   |   |   |-- ✅ postcss.config.js
|   |   |   |   |-- ✅ tailwind.config.cjs
|   |   |   |   |-- ✅ tailwind.config.js
|   |   |   |   \-- ✅ tailwind.config.ts
|   |   |   |-- ✅ types/
|   |   |   |   |-- ✅ generated/
|   |   |   |   |   |-- ✅ .gitkeep
|   |   |   |   |   |-- ✅ colors.d.ts
|   |   |   |   |   |-- ✅ corePluginList.d.ts
|   |   |   |   |   \-- ✅ default-theme.d.ts
|   |   |   |   |-- ✅ config.d.ts
|   |   |   |   \-- ✅ index.d.ts
|   |   |   |-- ✅ base.css
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ colors.d.ts
|   |   |   |-- ✅ colors.js
|   |   |   |-- ✅ components.css
|   |   |   |-- ✅ defaultConfig.d.ts
|   |   |   |-- ✅ defaultConfig.js
|   |   |   |-- ✅ defaultTheme.d.ts
|   |   |   |-- ✅ defaultTheme.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ loadConfig.d.ts
|   |   |   |-- ✅ loadConfig.js
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ plugin.d.ts
|   |   |   |-- ✅ plugin.js
|   |   |   |-- ✅ prettier.config.js
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ resolveConfig.d.ts
|   |   |   |-- ✅ resolveConfig.js
|   |   |   |-- ✅ screens.css
|   |   |   |-- ✅ tailwind.css
|   |   |   |-- ✅ utilities.css
|   |   |   \-- ✅ variants.css
|   |   |-- ✅ thenify/
|   |   |   |-- ✅ History.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ thenify-all/
|   |   |   |-- ✅ History.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ tinybench/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ index.cjs
|   |   |   |   |-- ✅ index.d.cts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ tinyexec/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ main.cjs
|   |   |   |   |-- ✅ main.d.cts
|   |   |   |   |-- ✅ main.d.ts
|   |   |   |   \-- ✅ main.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ tinyglobby/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ index.cjs
|   |   |   |   |-- ✅ index.d.cts
|   |   |   |   |-- ✅ index.d.mts
|   |   |   |   \-- ✅ index.mjs
|   |   |   |-- ✅ node_modules/
|   |   |   |   |-- ✅ fdir/
|   |   |   |   |   |-- ✅ dist/
|   |   |   |   |   |   |-- ✅ index.cjs
|   |   |   |   |   |   |-- ✅ index.d.cts
|   |   |   |   |   |   |-- ✅ index.d.mts
|   |   |   |   |   |   \-- ✅ index.mjs
|   |   |   |   |   |-- ✅ LICENSE
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   \-- ✅ README.md
|   |   |   |   \-- ✅ picomatch/
|   |   |   |       |-- ✅ lib/
|   |   |   |       |   |-- ✅ constants.js
|   |   |   |       |   |-- ✅ parse.js
|   |   |   |       |   |-- ✅ picomatch.js
|   |   |   |       |   |-- ✅ scan.js
|   |   |   |       |   \-- ✅ utils.js
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ LICENSE
|   |   |   |       |-- ✅ package.json
|   |   |   |       |-- ✅ posix.js
|   |   |   |       \-- ✅ README.md
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ tinypool/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ entry/
|   |   |   |   |   |-- ✅ process.d.ts
|   |   |   |   |   |-- ✅ process.js
|   |   |   |   |   |-- ✅ utils.d.ts
|   |   |   |   |   |-- ✅ utils.js
|   |   |   |   |   |-- ✅ worker.d.ts
|   |   |   |   |   \-- ✅ worker.js
|   |   |   |   |-- ✅ common-Qw-RoVFD.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ utils-B--2TaWv.js
|   |   |   |   \-- ✅ utils-De75vAgL.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ tinyrainbow/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ browser.d.ts
|   |   |   |   |-- ✅ browser.js
|   |   |   |   |-- ✅ chunk-BVHSVHOK.js
|   |   |   |   |-- ✅ index-8b61d5bc.d.ts
|   |   |   |   |-- ✅ node.d.ts
|   |   |   |   \-- ✅ node.js
|   |   |   |-- ✅ LICENCE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ tinyspy/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ LICENCE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ to-regex-range/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ tough-cookie/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ cookie.js
|   |   |   |   |-- ✅ memstore.js
|   |   |   |   |-- ✅ pathMatch.js
|   |   |   |   |-- ✅ permuteDomain.js
|   |   |   |   |-- ✅ pubsuffix-psl.js
|   |   |   |   |-- ✅ store.js
|   |   |   |   |-- ✅ utilHelper.js
|   |   |   |   |-- ✅ validators.js
|   |   |   |   \-- ✅ version.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ tr46/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ mappingTable.json
|   |   |   |   |-- ✅ regexes.js
|   |   |   |   \-- ✅ statusMapping.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ ts-interface-checker/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ types.d.ts
|   |   |   |   |-- ✅ types.js
|   |   |   |   |-- ✅ util.d.ts
|   |   |   |   \-- ✅ util.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ typescript/
|   |   |   |-- ✅ bin/
|   |   |   |   |-- ✅ tsc
|   |   |   |   \-- ✅ tsserver
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ cs/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ de/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ es/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ fr/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ it/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ ja/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ ko/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ pl/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ pt-br/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ ru/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ tr/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ zh-cn/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ zh-tw/
|   |   |   |   |   \-- ✅ diagnosticMessages.generated.json
|   |   |   |   |-- ✅ _tsc.js
|   |   |   |   |-- ✅ _tsserver.js
|   |   |   |   |-- ✅ _typingsInstaller.js
|   |   |   |   |-- ✅ lib.d.ts
|   |   |   |   |-- ✅ lib.decorators.d.ts
|   |   |   |   |-- ✅ lib.decorators.legacy.d.ts
|   |   |   |   |-- ✅ lib.dom.asynciterable.d.ts
|   |   |   |   |-- ✅ lib.dom.d.ts
|   |   |   |   |-- ✅ lib.dom.iterable.d.ts
|   |   |   |   |-- ✅ lib.es2015.collection.d.ts
|   |   |   |   |-- ✅ lib.es2015.core.d.ts
|   |   |   |   |-- ✅ lib.es2015.d.ts
|   |   |   |   |-- ✅ lib.es2015.generator.d.ts
|   |   |   |   |-- ✅ lib.es2015.iterable.d.ts
|   |   |   |   |-- ✅ lib.es2015.promise.d.ts
|   |   |   |   |-- ✅ lib.es2015.proxy.d.ts
|   |   |   |   |-- ✅ lib.es2015.reflect.d.ts
|   |   |   |   |-- ✅ lib.es2015.symbol.d.ts
|   |   |   |   |-- ✅ lib.es2015.symbol.wellknown.d.ts
|   |   |   |   |-- ✅ lib.es2016.array.include.d.ts
|   |   |   |   |-- ✅ lib.es2016.d.ts
|   |   |   |   |-- ✅ lib.es2016.full.d.ts
|   |   |   |   |-- ✅ lib.es2016.intl.d.ts
|   |   |   |   |-- ✅ lib.es2017.arraybuffer.d.ts
|   |   |   |   |-- ✅ lib.es2017.d.ts
|   |   |   |   |-- ✅ lib.es2017.date.d.ts
|   |   |   |   |-- ✅ lib.es2017.full.d.ts
|   |   |   |   |-- ✅ lib.es2017.intl.d.ts
|   |   |   |   |-- ✅ lib.es2017.object.d.ts
|   |   |   |   |-- ✅ lib.es2017.sharedmemory.d.ts
|   |   |   |   |-- ✅ lib.es2017.string.d.ts
|   |   |   |   |-- ✅ lib.es2017.typedarrays.d.ts
|   |   |   |   |-- ✅ lib.es2018.asyncgenerator.d.ts
|   |   |   |   |-- ✅ lib.es2018.asynciterable.d.ts
|   |   |   |   |-- ✅ lib.es2018.d.ts
|   |   |   |   |-- ✅ lib.es2018.full.d.ts
|   |   |   |   |-- ✅ lib.es2018.intl.d.ts
|   |   |   |   |-- ✅ lib.es2018.promise.d.ts
|   |   |   |   |-- ✅ lib.es2018.regexp.d.ts
|   |   |   |   |-- ✅ lib.es2019.array.d.ts
|   |   |   |   |-- ✅ lib.es2019.d.ts
|   |   |   |   |-- ✅ lib.es2019.full.d.ts
|   |   |   |   |-- ✅ lib.es2019.intl.d.ts
|   |   |   |   |-- ✅ lib.es2019.object.d.ts
|   |   |   |   |-- ✅ lib.es2019.string.d.ts
|   |   |   |   |-- ✅ lib.es2019.symbol.d.ts
|   |   |   |   |-- ✅ lib.es2020.bigint.d.ts
|   |   |   |   |-- ✅ lib.es2020.d.ts
|   |   |   |   |-- ✅ lib.es2020.date.d.ts
|   |   |   |   |-- ✅ lib.es2020.full.d.ts
|   |   |   |   |-- ✅ lib.es2020.intl.d.ts
|   |   |   |   |-- ✅ lib.es2020.number.d.ts
|   |   |   |   |-- ✅ lib.es2020.promise.d.ts
|   |   |   |   |-- ✅ lib.es2020.sharedmemory.d.ts
|   |   |   |   |-- ✅ lib.es2020.string.d.ts
|   |   |   |   |-- ✅ lib.es2020.symbol.wellknown.d.ts
|   |   |   |   |-- ✅ lib.es2021.d.ts
|   |   |   |   |-- ✅ lib.es2021.full.d.ts
|   |   |   |   |-- ✅ lib.es2021.intl.d.ts
|   |   |   |   |-- ✅ lib.es2021.promise.d.ts
|   |   |   |   |-- ✅ lib.es2021.string.d.ts
|   |   |   |   |-- ✅ lib.es2021.weakref.d.ts
|   |   |   |   |-- ✅ lib.es2022.array.d.ts
|   |   |   |   |-- ✅ lib.es2022.d.ts
|   |   |   |   |-- ✅ lib.es2022.error.d.ts
|   |   |   |   |-- ✅ lib.es2022.full.d.ts
|   |   |   |   |-- ✅ lib.es2022.intl.d.ts
|   |   |   |   |-- ✅ lib.es2022.object.d.ts
|   |   |   |   |-- ✅ lib.es2022.regexp.d.ts
|   |   |   |   |-- ✅ lib.es2022.string.d.ts
|   |   |   |   |-- ✅ lib.es2023.array.d.ts
|   |   |   |   |-- ✅ lib.es2023.collection.d.ts
|   |   |   |   |-- ✅ lib.es2023.d.ts
|   |   |   |   |-- ✅ lib.es2023.full.d.ts
|   |   |   |   |-- ✅ lib.es2023.intl.d.ts
|   |   |   |   |-- ✅ lib.es2024.arraybuffer.d.ts
|   |   |   |   |-- ✅ lib.es2024.collection.d.ts
|   |   |   |   |-- ✅ lib.es2024.d.ts
|   |   |   |   |-- ✅ lib.es2024.full.d.ts
|   |   |   |   |-- ✅ lib.es2024.object.d.ts
|   |   |   |   |-- ✅ lib.es2024.promise.d.ts
|   |   |   |   |-- ✅ lib.es2024.regexp.d.ts
|   |   |   |   |-- ✅ lib.es2024.sharedmemory.d.ts
|   |   |   |   |-- ✅ lib.es2024.string.d.ts
|   |   |   |   |-- ✅ lib.es5.d.ts
|   |   |   |   |-- ✅ lib.es6.d.ts
|   |   |   |   |-- ✅ lib.esnext.array.d.ts
|   |   |   |   |-- ✅ lib.esnext.collection.d.ts
|   |   |   |   |-- ✅ lib.esnext.d.ts
|   |   |   |   |-- ✅ lib.esnext.decorators.d.ts
|   |   |   |   |-- ✅ lib.esnext.disposable.d.ts
|   |   |   |   |-- ✅ lib.esnext.error.d.ts
|   |   |   |   |-- ✅ lib.esnext.float16.d.ts
|   |   |   |   |-- ✅ lib.esnext.full.d.ts
|   |   |   |   |-- ✅ lib.esnext.intl.d.ts
|   |   |   |   |-- ✅ lib.esnext.iterator.d.ts
|   |   |   |   |-- ✅ lib.esnext.promise.d.ts
|   |   |   |   |-- ✅ lib.esnext.sharedmemory.d.ts
|   |   |   |   |-- ✅ lib.scripthost.d.ts
|   |   |   |   |-- ✅ lib.webworker.asynciterable.d.ts
|   |   |   |   |-- ✅ lib.webworker.d.ts
|   |   |   |   |-- ✅ lib.webworker.importscripts.d.ts
|   |   |   |   |-- ✅ lib.webworker.iterable.d.ts
|   |   |   |   |-- ✅ tsc.js
|   |   |   |   |-- ✅ tsserver.js
|   |   |   |   |-- ✅ tsserverlibrary.d.ts
|   |   |   |   |-- ✅ tsserverlibrary.js
|   |   |   |   |-- ✅ typescript.d.ts
|   |   |   |   |-- ✅ typescript.js
|   |   |   |   |-- ✅ typesMap.json
|   |   |   |   |-- ✅ typingsInstaller.js
|   |   |   |   \-- ✅ watchGuard.js
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ SECURITY.md
|   |   |   \-- ✅ ThirdPartyNoticeText.txt
|   |   |-- ✅ undici-types/
|   |   |   |-- ✅ agent.d.ts
|   |   |   |-- ✅ api.d.ts
|   |   |   |-- ✅ balanced-pool.d.ts
|   |   |   |-- ✅ cache.d.ts
|   |   |   |-- ✅ client.d.ts
|   |   |   |-- ✅ connector.d.ts
|   |   |   |-- ✅ content-type.d.ts
|   |   |   |-- ✅ cookies.d.ts
|   |   |   |-- ✅ diagnostics-channel.d.ts
|   |   |   |-- ✅ dispatcher.d.ts
|   |   |   |-- ✅ env-http-proxy-agent.d.ts
|   |   |   |-- ✅ errors.d.ts
|   |   |   |-- ✅ eventsource.d.ts
|   |   |   |-- ✅ fetch.d.ts
|   |   |   |-- ✅ file.d.ts
|   |   |   |-- ✅ filereader.d.ts
|   |   |   |-- ✅ formdata.d.ts
|   |   |   |-- ✅ global-dispatcher.d.ts
|   |   |   |-- ✅ global-origin.d.ts
|   |   |   |-- ✅ handlers.d.ts
|   |   |   |-- ✅ header.d.ts
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ interceptors.d.ts
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ mock-agent.d.ts
|   |   |   |-- ✅ mock-client.d.ts
|   |   |   |-- ✅ mock-errors.d.ts
|   |   |   |-- ✅ mock-interceptor.d.ts
|   |   |   |-- ✅ mock-pool.d.ts
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ patch.d.ts
|   |   |   |-- ✅ pool-stats.d.ts
|   |   |   |-- ✅ pool.d.ts
|   |   |   |-- ✅ proxy-agent.d.ts
|   |   |   |-- ✅ readable.d.ts
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ retry-agent.d.ts
|   |   |   |-- ✅ retry-handler.d.ts
|   |   |   |-- ✅ util.d.ts
|   |   |   |-- ✅ webidl.d.ts
|   |   |   \-- ✅ websocket.d.ts
|   |   |-- ✅ universalify/
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ update-browserslist-db/
|   |   |   |-- ✅ check-npm-version.js
|   |   |   |-- ✅ cli.js
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ utils.js
|   |   |-- ✅ url-parse/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ url-parse.js
|   |   |   |   |-- ✅ url-parse.min.js
|   |   |   |   \-- ✅ url-parse.min.js.map
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ util-deprecate/
|   |   |   |-- ✅ browser.js
|   |   |   |-- ✅ History.md
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ node.js
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ vite/
|   |   |   |-- ✅ bin/
|   |   |   |   |-- ✅ openChrome.js
|   |   |   |   \-- ✅ vite.js
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ client/
|   |   |   |   |   |-- ✅ client.mjs
|   |   |   |   |   \-- ✅ env.mjs
|   |   |   |   \-- ✅ node/
|   |   |   |       |-- ✅ chunks/
|   |   |   |       |   |-- ✅ dep-BbmkDZt5.js
|   |   |   |       |   |-- ✅ dep-BFcSm8xQ.js
|   |   |   |       |   |-- ✅ dep-Bm2ujbhY.js
|   |   |   |       |   |-- ✅ dep-BRWmquJk.js
|   |   |   |       |   |-- ✅ dep-BuoK8Wda.js
|   |   |   |       |   |-- ✅ dep-CAc8-XM0.js
|   |   |   |       |   |-- ✅ dep-CCSnTAeo.js
|   |   |   |       |   |-- ✅ dep-CwrJo3zV.js
|   |   |   |       |   |-- ✅ dep-D8ZQhg7-.js
|   |   |   |       |   |-- ✅ dep-H0AnFej7.js
|   |   |   |       |   |-- ✅ dep-lCKrEJQm.js
|   |   |   |       |   \-- ✅ dep-SmwnYDP9.js
|   |   |   |       |-- ✅ cli.js
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ module-runner.d.ts
|   |   |   |       |-- ✅ module-runner.js
|   |   |   |       \-- ✅ moduleRunnerTransport-BWUZBVLX.d.ts
|   |   |   |-- ✅ misc/
|   |   |   |   |-- ✅ false.js
|   |   |   |   \-- ✅ true.js
|   |   |   |-- ✅ node_modules/
|   |   |   |   |-- ✅ fdir/
|   |   |   |   |   |-- ✅ dist/
|   |   |   |   |   |   |-- ✅ index.cjs
|   |   |   |   |   |   |-- ✅ index.d.cts
|   |   |   |   |   |   |-- ✅ index.d.mts
|   |   |   |   |   |   \-- ✅ index.mjs
|   |   |   |   |   |-- ✅ LICENSE
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   \-- ✅ README.md
|   |   |   |   \-- ✅ picomatch/
|   |   |   |       |-- ✅ lib/
|   |   |   |       |   |-- ✅ constants.js
|   |   |   |       |   |-- ✅ parse.js
|   |   |   |       |   |-- ✅ picomatch.js
|   |   |   |       |   |-- ✅ scan.js
|   |   |   |       |   \-- ✅ utils.js
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ LICENSE
|   |   |   |       |-- ✅ package.json
|   |   |   |       |-- ✅ posix.js
|   |   |   |       \-- ✅ README.md
|   |   |   |-- ✅ types/
|   |   |   |   |-- ✅ internal/
|   |   |   |   |   |-- ✅ cssPreprocessorOptions.d.ts
|   |   |   |   |   |-- ✅ lightningcssOptions.d.ts
|   |   |   |   |   \-- ✅ terserOptions.d.ts
|   |   |   |   |-- ✅ customEvent.d.ts
|   |   |   |   |-- ✅ hmrPayload.d.ts
|   |   |   |   |-- ✅ hot.d.ts
|   |   |   |   |-- ✅ import-meta.d.ts
|   |   |   |   |-- ✅ importGlob.d.ts
|   |   |   |   |-- ✅ importMeta.d.ts
|   |   |   |   |-- ✅ metadata.d.ts
|   |   |   |   \-- ✅ package.json
|   |   |   |-- ✅ client.d.ts
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ vite-node/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ chunk-browser.cjs
|   |   |   |   |-- ✅ chunk-browser.mjs
|   |   |   |   |-- ✅ chunk-hmr.cjs
|   |   |   |   |-- ✅ chunk-hmr.mjs
|   |   |   |   |-- ✅ cli.cjs
|   |   |   |   |-- ✅ cli.d.ts
|   |   |   |   |-- ✅ cli.mjs
|   |   |   |   |-- ✅ client.cjs
|   |   |   |   |-- ✅ client.d.ts
|   |   |   |   |-- ✅ client.mjs
|   |   |   |   |-- ✅ constants.cjs
|   |   |   |   |-- ✅ constants.d.ts
|   |   |   |   |-- ✅ constants.mjs
|   |   |   |   |-- ✅ hmr.cjs
|   |   |   |   |-- ✅ hmr.d.ts
|   |   |   |   |-- ✅ hmr.mjs
|   |   |   |   |-- ✅ index.cjs
|   |   |   |   |-- ✅ index.d-DGmxD2U7.d.ts
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.mjs
|   |   |   |   |-- ✅ server.cjs
|   |   |   |   |-- ✅ server.d.ts
|   |   |   |   |-- ✅ server.mjs
|   |   |   |   |-- ✅ source-map.cjs
|   |   |   |   |-- ✅ source-map.d.ts
|   |   |   |   |-- ✅ source-map.mjs
|   |   |   |   |-- ✅ trace-mapping.d-DLVdEqOp.d.ts
|   |   |   |   |-- ✅ types.cjs
|   |   |   |   |-- ✅ types.d.ts
|   |   |   |   |-- ✅ types.mjs
|   |   |   |   |-- ✅ utils.cjs
|   |   |   |   |-- ✅ utils.d.ts
|   |   |   |   \-- ✅ utils.mjs
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ vite-node.mjs
|   |   |-- ✅ vitest/
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ chunks/
|   |   |   |   |   |-- ✅ _commonjsHelpers.BFTU3MAI.js
|   |   |   |   |   |-- ✅ base.DfmxU-tU.js
|   |   |   |   |   |-- ✅ benchmark.CYdenmiT.js
|   |   |   |   |   |-- ✅ benchmark.d.BwvBVTda.d.ts
|   |   |   |   |   |-- ✅ cac.Cb-PYCCB.js
|   |   |   |   |   |-- ✅ cli-api.BkDphVBG.js
|   |   |   |   |   |-- ✅ config.d.D2ROskhv.d.ts
|   |   |   |   |   |-- ✅ console.CtFJOzRO.js
|   |   |   |   |   |-- ✅ constants.DnKduX2e.js
|   |   |   |   |   |-- ✅ coverage.d.S9RMNXIe.d.ts
|   |   |   |   |   |-- ✅ coverage.DL5VHqXY.js
|   |   |   |   |   |-- ✅ coverage.DVF1vEu8.js
|   |   |   |   |   |-- ✅ creator.GK6I-cL4.js
|   |   |   |   |   |-- ✅ date.Bq6ZW5rf.js
|   |   |   |   |   |-- ✅ defaults.B7q_naMc.js
|   |   |   |   |   |-- ✅ env.D4Lgay0q.js
|   |   |   |   |   |-- ✅ environment.d.cL3nLXbE.d.ts
|   |   |   |   |   |-- ✅ execute.B7h3T_Hc.js
|   |   |   |   |   |-- ✅ git.BVQ8w_Sw.js
|   |   |   |   |   |-- ✅ global.d.MAmajcmJ.d.ts
|   |   |   |   |   |-- ✅ globals.DEHgCU4V.js
|   |   |   |   |   |-- ✅ index.B521nVV-.js
|   |   |   |   |   |-- ✅ index.BCWujgDG.js
|   |   |   |   |   |-- ✅ index.CdQS2e2Q.js
|   |   |   |   |   |-- ✅ index.CmSc2RE5.js
|   |   |   |   |   |-- ✅ index.CwejwG0H.js
|   |   |   |   |   |-- ✅ index.D3XRDfWc.js
|   |   |   |   |   |-- ✅ index.VByaPkjc.js
|   |   |   |   |   |-- ✅ index.X0nbfr6-.js
|   |   |   |   |   |-- ✅ inspector.C914Efll.js
|   |   |   |   |   |-- ✅ mocker.d.BE_2ls6u.d.ts
|   |   |   |   |   |-- ✅ node.fjCdwEIl.js
|   |   |   |   |   |-- ✅ reporters.d.BFLkQcL6.d.ts
|   |   |   |   |   |-- ✅ rpc.-pEldfrD.js
|   |   |   |   |   |-- ✅ runBaseTests.9Ij9_de-.js
|   |   |   |   |   |-- ✅ setup-common.Dd054P77.js
|   |   |   |   |   |-- ✅ suite.d.FvehnV49.d.ts
|   |   |   |   |   |-- ✅ typechecker.DRKU1-1g.js
|   |   |   |   |   |-- ✅ utils.CAioKnHs.js
|   |   |   |   |   |-- ✅ utils.XdZDrNZV.js
|   |   |   |   |   |-- ✅ vi.bdSIJ99Y.js
|   |   |   |   |   |-- ✅ vite.d.CMLlLIFP.d.ts
|   |   |   |   |   |-- ✅ vm.BThCzidc.js
|   |   |   |   |   |-- ✅ worker.d.1GmBbd7G.d.ts
|   |   |   |   |   \-- ✅ worker.d.CKwWzBSj.d.ts
|   |   |   |   |-- ✅ workers/
|   |   |   |   |   |-- ✅ forks.js
|   |   |   |   |   |-- ✅ runVmTests.js
|   |   |   |   |   |-- ✅ threads.js
|   |   |   |   |   |-- ✅ vmForks.js
|   |   |   |   |   \-- ✅ vmThreads.js
|   |   |   |   |-- ✅ browser.d.ts
|   |   |   |   |-- ✅ browser.js
|   |   |   |   |-- ✅ cli.js
|   |   |   |   |-- ✅ config.cjs
|   |   |   |   |-- ✅ config.d.ts
|   |   |   |   |-- ✅ config.js
|   |   |   |   |-- ✅ coverage.d.ts
|   |   |   |   |-- ✅ coverage.js
|   |   |   |   |-- ✅ environments.d.ts
|   |   |   |   |-- ✅ environments.js
|   |   |   |   |-- ✅ execute.d.ts
|   |   |   |   |-- ✅ execute.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ mocker.d.ts
|   |   |   |   |-- ✅ mocker.js
|   |   |   |   |-- ✅ node.d.ts
|   |   |   |   |-- ✅ node.js
|   |   |   |   |-- ✅ path.js
|   |   |   |   |-- ✅ reporters.d.ts
|   |   |   |   |-- ✅ reporters.js
|   |   |   |   |-- ✅ runners.d.ts
|   |   |   |   |-- ✅ runners.js
|   |   |   |   |-- ✅ snapshot.d.ts
|   |   |   |   |-- ✅ snapshot.js
|   |   |   |   |-- ✅ spy.js
|   |   |   |   |-- ✅ suite.d.ts
|   |   |   |   |-- ✅ suite.js
|   |   |   |   |-- ✅ worker.js
|   |   |   |   |-- ✅ workers.d.ts
|   |   |   |   \-- ✅ workers.js
|   |   |   |-- ✅ node_modules/
|   |   |   |   \-- ✅ picomatch/
|   |   |   |       |-- ✅ lib/
|   |   |   |       |   |-- ✅ constants.js
|   |   |   |       |   |-- ✅ parse.js
|   |   |   |       |   |-- ✅ picomatch.js
|   |   |   |       |   |-- ✅ scan.js
|   |   |   |       |   \-- ✅ utils.js
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ LICENSE
|   |   |   |       |-- ✅ package.json
|   |   |   |       |-- ✅ posix.js
|   |   |   |       \-- ✅ README.md
|   |   |   |-- ✅ browser.d.ts
|   |   |   |-- ✅ config.d.ts
|   |   |   |-- ✅ coverage.d.ts
|   |   |   |-- ✅ environments.d.ts
|   |   |   |-- ✅ execute.d.ts
|   |   |   |-- ✅ globals.d.ts
|   |   |   |-- ✅ import-meta.d.ts
|   |   |   |-- ✅ importMeta.d.ts
|   |   |   |-- ✅ index.cjs
|   |   |   |-- ✅ index.d.cts
|   |   |   |-- ✅ jsdom.d.ts
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ mocker.d.ts
|   |   |   |-- ✅ node.d.ts
|   |   |   |-- ✅ optional-types.d.ts
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ reporters.d.ts
|   |   |   |-- ✅ runners.d.ts
|   |   |   |-- ✅ snapshot.d.ts
|   |   |   |-- ✅ suite.d.ts
|   |   |   |-- ✅ suppress-warnings.cjs
|   |   |   |-- ✅ utils.d.ts
|   |   |   |-- ✅ vitest.mjs
|   |   |   \-- ✅ workers.d.ts
|   |   |-- ✅ w3c-xmlserializer/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ attributes.js
|   |   |   |   |-- ✅ constants.js
|   |   |   |   \-- ✅ serialize.js
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ webidl-conversions/
|   |   |   |-- ✅ lib/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ LICENSE.md
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ whatwg-encoding/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ labels-to-names.json
|   |   |   |   |-- ✅ supported-names.json
|   |   |   |   \-- ✅ whatwg-encoding.js
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ whatwg-mimetype/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ mime-type-parameters.js
|   |   |   |   |-- ✅ mime-type.js
|   |   |   |   |-- ✅ parser.js
|   |   |   |   |-- ✅ serializer.js
|   |   |   |   \-- ✅ utils.js
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ whatwg-url/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ encoding.js
|   |   |   |   |-- ✅ Function.js
|   |   |   |   |-- ✅ infra.js
|   |   |   |   |-- ✅ percent-encoding.js
|   |   |   |   |-- ✅ URL-impl.js
|   |   |   |   |-- ✅ url-state-machine.js
|   |   |   |   |-- ✅ URL.js
|   |   |   |   |-- ✅ urlencoded.js
|   |   |   |   |-- ✅ URLSearchParams-impl.js
|   |   |   |   |-- ✅ URLSearchParams.js
|   |   |   |   |-- ✅ utils.js
|   |   |   |   \-- ✅ VoidFunction.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ webidl2js-wrapper.js
|   |   |-- ✅ which/
|   |   |   |-- ✅ bin/
|   |   |   |   \-- ✅ node-which
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ which.js
|   |   |-- ✅ which-boxed-primitive/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ which-collection/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ which-typed-array/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ test/
|   |   |   |   \-- ✅ index.js
|   |   |   |-- ✅ .editorconfig
|   |   |   |-- ✅ .eslintrc
|   |   |   |-- ✅ .nycrc
|   |   |   |-- ✅ CHANGELOG.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ tsconfig.json
|   |   |-- ✅ why-is-node-running/
|   |   |   |-- ✅ .github/
|   |   |   |   \-- ✅ FUNDING.yml
|   |   |   |-- ✅ cli.js
|   |   |   |-- ✅ example.js
|   |   |   |-- ✅ include.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ wrap-ansi/
|   |   |   |-- ✅ node_modules/
|   |   |   |   \-- ✅ ansi-styles/
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ license
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ readme.md
|   |   |   |-- ✅ index.d.ts
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ wrap-ansi-cjs/
|   |   |   |-- ✅ node_modules/
|   |   |   |   |-- ✅ emoji-regex/
|   |   |   |   |   |-- ✅ es2015/
|   |   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |   \-- ✅ text.js
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ LICENSE-MIT.txt
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   |-- ✅ README.md
|   |   |   |   |   \-- ✅ text.js
|   |   |   |   |-- ✅ string-width/
|   |   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ license
|   |   |   |   |   |-- ✅ package.json
|   |   |   |   |   \-- ✅ readme.md
|   |   |   |   \-- ✅ strip-ansi/
|   |   |   |       |-- ✅ index.d.ts
|   |   |   |       |-- ✅ index.js
|   |   |   |       |-- ✅ license
|   |   |   |       |-- ✅ package.json
|   |   |   |       \-- ✅ readme.md
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ license
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ readme.md
|   |   |-- ✅ ws/
|   |   |   |-- ✅ lib/
|   |   |   |   |-- ✅ buffer-util.js
|   |   |   |   |-- ✅ constants.js
|   |   |   |   |-- ✅ event-target.js
|   |   |   |   |-- ✅ extension.js
|   |   |   |   |-- ✅ limiter.js
|   |   |   |   |-- ✅ permessage-deflate.js
|   |   |   |   |-- ✅ receiver.js
|   |   |   |   |-- ✅ sender.js
|   |   |   |   |-- ✅ stream.js
|   |   |   |   |-- ✅ subprotocol.js
|   |   |   |   |-- ✅ validation.js
|   |   |   |   |-- ✅ websocket-server.js
|   |   |   |   \-- ✅ websocket.js
|   |   |   |-- ✅ browser.js
|   |   |   |-- ✅ index.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ wrapper.mjs
|   |   |-- ✅ xml-name-validator/
|   |   |   |-- ✅ lib/
|   |   |   |   \-- ✅ xml-name-validator.js
|   |   |   |-- ✅ LICENSE.txt
|   |   |   |-- ✅ package.json
|   |   |   \-- ✅ README.md
|   |   |-- ✅ xmlchars/
|   |   |   |-- ✅ xml/
|   |   |   |   |-- ✅ 1.0/
|   |   |   |   |   |-- ✅ ed4.d.ts
|   |   |   |   |   |-- ✅ ed4.js
|   |   |   |   |   |-- ✅ ed4.js.map
|   |   |   |   |   |-- ✅ ed5.d.ts
|   |   |   |   |   |-- ✅ ed5.js
|   |   |   |   |   \-- ✅ ed5.js.map
|   |   |   |   \-- ✅ 1.1/
|   |   |   |       |-- ✅ ed2.d.ts
|   |   |   |       |-- ✅ ed2.js
|   |   |   |       \-- ✅ ed2.js.map
|   |   |   |-- ✅ xmlns/
|   |   |   |   \-- ✅ 1.0/
|   |   |   |       |-- ✅ ed3.d.ts
|   |   |   |       |-- ✅ ed3.js
|   |   |   |       \-- ✅ ed3.js.map
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   |-- ✅ xmlchars.d.ts
|   |   |   |-- ✅ xmlchars.js
|   |   |   \-- ✅ xmlchars.js.map
|   |   |-- ✅ yallist/
|   |   |   |-- ✅ iterator.js
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ yallist.js
|   |   |-- ✅ yaml/
|   |   |   |-- ✅ browser/
|   |   |   |   |-- ✅ dist/
|   |   |   |   |   |-- ✅ compose/
|   |   |   |   |   |   |-- ✅ compose-collection.js
|   |   |   |   |   |   |-- ✅ compose-doc.js
|   |   |   |   |   |   |-- ✅ compose-node.js
|   |   |   |   |   |   |-- ✅ compose-scalar.js
|   |   |   |   |   |   |-- ✅ composer.js
|   |   |   |   |   |   |-- ✅ resolve-block-map.js
|   |   |   |   |   |   |-- ✅ resolve-block-scalar.js
|   |   |   |   |   |   |-- ✅ resolve-block-seq.js
|   |   |   |   |   |   |-- ✅ resolve-end.js
|   |   |   |   |   |   |-- ✅ resolve-flow-collection.js
|   |   |   |   |   |   |-- ✅ resolve-flow-scalar.js
|   |   |   |   |   |   |-- ✅ resolve-props.js
|   |   |   |   |   |   |-- ✅ util-contains-newline.js
|   |   |   |   |   |   |-- ✅ util-empty-scalar-position.js
|   |   |   |   |   |   |-- ✅ util-flow-indent-check.js
|   |   |   |   |   |   \-- ✅ util-map-includes.js
|   |   |   |   |   |-- ✅ doc/
|   |   |   |   |   |   |-- ✅ anchors.js
|   |   |   |   |   |   |-- ✅ applyReviver.js
|   |   |   |   |   |   |-- ✅ createNode.js
|   |   |   |   |   |   |-- ✅ directives.js
|   |   |   |   |   |   \-- ✅ Document.js
|   |   |   |   |   |-- ✅ nodes/
|   |   |   |   |   |   |-- ✅ addPairToJSMap.js
|   |   |   |   |   |   |-- ✅ Alias.js
|   |   |   |   |   |   |-- ✅ Collection.js
|   |   |   |   |   |   |-- ✅ identity.js
|   |   |   |   |   |   |-- ✅ Node.js
|   |   |   |   |   |   |-- ✅ Pair.js
|   |   |   |   |   |   |-- ✅ Scalar.js
|   |   |   |   |   |   |-- ✅ toJS.js
|   |   |   |   |   |   |-- ✅ YAMLMap.js
|   |   |   |   |   |   \-- ✅ YAMLSeq.js
|   |   |   |   |   |-- ✅ parse/
|   |   |   |   |   |   |-- ✅ cst-scalar.js
|   |   |   |   |   |   |-- ✅ cst-stringify.js
|   |   |   |   |   |   |-- ✅ cst-visit.js
|   |   |   |   |   |   |-- ✅ cst.js
|   |   |   |   |   |   |-- ✅ lexer.js
|   |   |   |   |   |   |-- ✅ line-counter.js
|   |   |   |   |   |   \-- ✅ parser.js
|   |   |   |   |   |-- ✅ schema/
|   |   |   |   |   |   |-- ✅ common/
|   |   |   |   |   |   |   |-- ✅ map.js
|   |   |   |   |   |   |   |-- ✅ null.js
|   |   |   |   |   |   |   |-- ✅ seq.js
|   |   |   |   |   |   |   \-- ✅ string.js
|   |   |   |   |   |   |-- ✅ core/
|   |   |   |   |   |   |   |-- ✅ bool.js
|   |   |   |   |   |   |   |-- ✅ float.js
|   |   |   |   |   |   |   |-- ✅ int.js
|   |   |   |   |   |   |   \-- ✅ schema.js
|   |   |   |   |   |   |-- ✅ json/
|   |   |   |   |   |   |   \-- ✅ schema.js
|   |   |   |   |   |   |-- ✅ yaml-1.1/
|   |   |   |   |   |   |   |-- ✅ binary.js
|   |   |   |   |   |   |   |-- ✅ bool.js
|   |   |   |   |   |   |   |-- ✅ float.js
|   |   |   |   |   |   |   |-- ✅ int.js
|   |   |   |   |   |   |   |-- ✅ merge.js
|   |   |   |   |   |   |   |-- ✅ omap.js
|   |   |   |   |   |   |   |-- ✅ pairs.js
|   |   |   |   |   |   |   |-- ✅ schema.js
|   |   |   |   |   |   |   |-- ✅ set.js
|   |   |   |   |   |   |   \-- ✅ timestamp.js
|   |   |   |   |   |   |-- ✅ Schema.js
|   |   |   |   |   |   \-- ✅ tags.js
|   |   |   |   |   |-- ✅ stringify/
|   |   |   |   |   |   |-- ✅ foldFlowLines.js
|   |   |   |   |   |   |-- ✅ stringify.js
|   |   |   |   |   |   |-- ✅ stringifyCollection.js
|   |   |   |   |   |   |-- ✅ stringifyComment.js
|   |   |   |   |   |   |-- ✅ stringifyDocument.js
|   |   |   |   |   |   |-- ✅ stringifyNumber.js
|   |   |   |   |   |   |-- ✅ stringifyPair.js
|   |   |   |   |   |   \-- ✅ stringifyString.js
|   |   |   |   |   |-- ✅ errors.js
|   |   |   |   |   |-- ✅ index.js
|   |   |   |   |   |-- ✅ log.js
|   |   |   |   |   |-- ✅ public-api.js
|   |   |   |   |   |-- ✅ util.js
|   |   |   |   |   \-- ✅ visit.js
|   |   |   |   |-- ✅ index.js
|   |   |   |   \-- ✅ package.json
|   |   |   |-- ✅ dist/
|   |   |   |   |-- ✅ compose/
|   |   |   |   |   |-- ✅ compose-collection.d.ts
|   |   |   |   |   |-- ✅ compose-collection.js
|   |   |   |   |   |-- ✅ compose-doc.d.ts
|   |   |   |   |   |-- ✅ compose-doc.js
|   |   |   |   |   |-- ✅ compose-node.d.ts
|   |   |   |   |   |-- ✅ compose-node.js
|   |   |   |   |   |-- ✅ compose-scalar.d.ts
|   |   |   |   |   |-- ✅ compose-scalar.js
|   |   |   |   |   |-- ✅ composer.d.ts
|   |   |   |   |   |-- ✅ composer.js
|   |   |   |   |   |-- ✅ resolve-block-map.d.ts
|   |   |   |   |   |-- ✅ resolve-block-map.js
|   |   |   |   |   |-- ✅ resolve-block-scalar.d.ts
|   |   |   |   |   |-- ✅ resolve-block-scalar.js
|   |   |   |   |   |-- ✅ resolve-block-seq.d.ts
|   |   |   |   |   |-- ✅ resolve-block-seq.js
|   |   |   |   |   |-- ✅ resolve-end.d.ts
|   |   |   |   |   |-- ✅ resolve-end.js
|   |   |   |   |   |-- ✅ resolve-flow-collection.d.ts
|   |   |   |   |   |-- ✅ resolve-flow-collection.js
|   |   |   |   |   |-- ✅ resolve-flow-scalar.d.ts
|   |   |   |   |   |-- ✅ resolve-flow-scalar.js
|   |   |   |   |   |-- ✅ resolve-props.d.ts
|   |   |   |   |   |-- ✅ resolve-props.js
|   |   |   |   |   |-- ✅ util-contains-newline.d.ts
|   |   |   |   |   |-- ✅ util-contains-newline.js
|   |   |   |   |   |-- ✅ util-empty-scalar-position.d.ts
|   |   |   |   |   |-- ✅ util-empty-scalar-position.js
|   |   |   |   |   |-- ✅ util-flow-indent-check.d.ts
|   |   |   |   |   |-- ✅ util-flow-indent-check.js
|   |   |   |   |   |-- ✅ util-map-includes.d.ts
|   |   |   |   |   \-- ✅ util-map-includes.js
|   |   |   |   |-- ✅ doc/
|   |   |   |   |   |-- ✅ anchors.d.ts
|   |   |   |   |   |-- ✅ anchors.js
|   |   |   |   |   |-- ✅ applyReviver.d.ts
|   |   |   |   |   |-- ✅ applyReviver.js
|   |   |   |   |   |-- ✅ createNode.d.ts
|   |   |   |   |   |-- ✅ createNode.js
|   |   |   |   |   |-- ✅ directives.d.ts
|   |   |   |   |   |-- ✅ directives.js
|   |   |   |   |   |-- ✅ Document.d.ts
|   |   |   |   |   \-- ✅ Document.js
|   |   |   |   |-- ✅ nodes/
|   |   |   |   |   |-- ✅ addPairToJSMap.d.ts
|   |   |   |   |   |-- ✅ addPairToJSMap.js
|   |   |   |   |   |-- ✅ Alias.d.ts
|   |   |   |   |   |-- ✅ Alias.js
|   |   |   |   |   |-- ✅ Collection.d.ts
|   |   |   |   |   |-- ✅ Collection.js
|   |   |   |   |   |-- ✅ identity.d.ts
|   |   |   |   |   |-- ✅ identity.js
|   |   |   |   |   |-- ✅ Node.d.ts
|   |   |   |   |   |-- ✅ Node.js
|   |   |   |   |   |-- ✅ Pair.d.ts
|   |   |   |   |   |-- ✅ Pair.js
|   |   |   |   |   |-- ✅ Scalar.d.ts
|   |   |   |   |   |-- ✅ Scalar.js
|   |   |   |   |   |-- ✅ toJS.d.ts
|   |   |   |   |   |-- ✅ toJS.js
|   |   |   |   |   |-- ✅ YAMLMap.d.ts
|   |   |   |   |   |-- ✅ YAMLMap.js
|   |   |   |   |   |-- ✅ YAMLSeq.d.ts
|   |   |   |   |   \-- ✅ YAMLSeq.js
|   |   |   |   |-- ✅ parse/
|   |   |   |   |   |-- ✅ cst-scalar.d.ts
|   |   |   |   |   |-- ✅ cst-scalar.js
|   |   |   |   |   |-- ✅ cst-stringify.d.ts
|   |   |   |   |   |-- ✅ cst-stringify.js
|   |   |   |   |   |-- ✅ cst-visit.d.ts
|   |   |   |   |   |-- ✅ cst-visit.js
|   |   |   |   |   |-- ✅ cst.d.ts
|   |   |   |   |   |-- ✅ cst.js
|   |   |   |   |   |-- ✅ lexer.d.ts
|   |   |   |   |   |-- ✅ lexer.js
|   |   |   |   |   |-- ✅ line-counter.d.ts
|   |   |   |   |   |-- ✅ line-counter.js
|   |   |   |   |   |-- ✅ parser.d.ts
|   |   |   |   |   \-- ✅ parser.js
|   |   |   |   |-- ✅ schema/
|   |   |   |   |   |-- ✅ common/
|   |   |   |   |   |   |-- ✅ map.d.ts
|   |   |   |   |   |   |-- ✅ map.js
|   |   |   |   |   |   |-- ✅ null.d.ts
|   |   |   |   |   |   |-- ✅ null.js
|   |   |   |   |   |   |-- ✅ seq.d.ts
|   |   |   |   |   |   |-- ✅ seq.js
|   |   |   |   |   |   |-- ✅ string.d.ts
|   |   |   |   |   |   \-- ✅ string.js
|   |   |   |   |   |-- ✅ core/
|   |   |   |   |   |   |-- ✅ bool.d.ts
|   |   |   |   |   |   |-- ✅ bool.js
|   |   |   |   |   |   |-- ✅ float.d.ts
|   |   |   |   |   |   |-- ✅ float.js
|   |   |   |   |   |   |-- ✅ int.d.ts
|   |   |   |   |   |   |-- ✅ int.js
|   |   |   |   |   |   |-- ✅ schema.d.ts
|   |   |   |   |   |   \-- ✅ schema.js
|   |   |   |   |   |-- ✅ json/
|   |   |   |   |   |   |-- ✅ schema.d.ts
|   |   |   |   |   |   \-- ✅ schema.js
|   |   |   |   |   |-- ✅ yaml-1.1/
|   |   |   |   |   |   |-- ✅ binary.d.ts
|   |   |   |   |   |   |-- ✅ binary.js
|   |   |   |   |   |   |-- ✅ bool.d.ts
|   |   |   |   |   |   |-- ✅ bool.js
|   |   |   |   |   |   |-- ✅ float.d.ts
|   |   |   |   |   |   |-- ✅ float.js
|   |   |   |   |   |   |-- ✅ int.d.ts
|   |   |   |   |   |   |-- ✅ int.js
|   |   |   |   |   |   |-- ✅ merge.d.ts
|   |   |   |   |   |   |-- ✅ merge.js
|   |   |   |   |   |   |-- ✅ omap.d.ts
|   |   |   |   |   |   |-- ✅ omap.js
|   |   |   |   |   |   |-- ✅ pairs.d.ts
|   |   |   |   |   |   |-- ✅ pairs.js
|   |   |   |   |   |   |-- ✅ schema.d.ts
|   |   |   |   |   |   |-- ✅ schema.js
|   |   |   |   |   |   |-- ✅ set.d.ts
|   |   |   |   |   |   |-- ✅ set.js
|   |   |   |   |   |   |-- ✅ timestamp.d.ts
|   |   |   |   |   |   \-- ✅ timestamp.js
|   |   |   |   |   |-- ✅ json-schema.d.ts
|   |   |   |   |   |-- ✅ Schema.d.ts
|   |   |   |   |   |-- ✅ Schema.js
|   |   |   |   |   |-- ✅ tags.d.ts
|   |   |   |   |   |-- ✅ tags.js
|   |   |   |   |   \-- ✅ types.d.ts
|   |   |   |   |-- ✅ stringify/
|   |   |   |   |   |-- ✅ foldFlowLines.d.ts
|   |   |   |   |   |-- ✅ foldFlowLines.js
|   |   |   |   |   |-- ✅ stringify.d.ts
|   |   |   |   |   |-- ✅ stringify.js
|   |   |   |   |   |-- ✅ stringifyCollection.d.ts
|   |   |   |   |   |-- ✅ stringifyCollection.js
|   |   |   |   |   |-- ✅ stringifyComment.d.ts
|   |   |   |   |   |-- ✅ stringifyComment.js
|   |   |   |   |   |-- ✅ stringifyDocument.d.ts
|   |   |   |   |   |-- ✅ stringifyDocument.js
|   |   |   |   |   |-- ✅ stringifyNumber.d.ts
|   |   |   |   |   |-- ✅ stringifyNumber.js
|   |   |   |   |   |-- ✅ stringifyPair.d.ts
|   |   |   |   |   |-- ✅ stringifyPair.js
|   |   |   |   |   |-- ✅ stringifyString.d.ts
|   |   |   |   |   \-- ✅ stringifyString.js
|   |   |   |   |-- ✅ cli.d.ts
|   |   |   |   |-- ✅ cli.mjs
|   |   |   |   |-- ✅ errors.d.ts
|   |   |   |   |-- ✅ errors.js
|   |   |   |   |-- ✅ index.d.ts
|   |   |   |   |-- ✅ index.js
|   |   |   |   |-- ✅ log.d.ts
|   |   |   |   |-- ✅ log.js
|   |   |   |   |-- ✅ options.d.ts
|   |   |   |   |-- ✅ public-api.d.ts
|   |   |   |   |-- ✅ public-api.js
|   |   |   |   |-- ✅ test-events.d.ts
|   |   |   |   |-- ✅ test-events.js
|   |   |   |   |-- ✅ util.d.ts
|   |   |   |   |-- ✅ util.js
|   |   |   |   |-- ✅ visit.d.ts
|   |   |   |   \-- ✅ visit.js
|   |   |   |-- ✅ bin.mjs
|   |   |   |-- ✅ LICENSE
|   |   |   |-- ✅ package.json
|   |   |   |-- ✅ README.md
|   |   |   \-- ✅ util.js
|   |   \-- ✅ .package-lock.json
|   |-- ✅ public/
|   |   \-- ✅ vite.svg
|   |-- ✅ src/
|   |   |-- ✅ __tests__/
|   |   |   |-- ✅ fallback.test.ts
|   |   |   |-- ✅ prices.test.ts
|   |   |   |-- ✅ safejson.test.ts
|   |   |   |-- ✅ Settings.setDefaultExchange.test.tsx
|   |   |   |-- ✅ Settings.test.tsx
|   |   |   |-- ✅ smoke.test.ts
|   |   |   |-- ✅ StatusCard.test.tsx
|   |   |   \-- ✅ ws.test.ts
|   |   |-- ✅ api/
|   |   |   |-- ✅ prices.ts
|   |   |   |-- ✅ signals.ts
|   |   |   \-- ✅ stress.ts
|   |   |-- ✅ assets/
|   |   |   \-- ✅ system-architecture.svg
|   |   |-- ✅ components/
|   |   |   |-- ✅ __tests__/
|   |   |   |   \-- ✅ PriceChart.test.tsx
|   |   |   |-- ✅ analysis/
|   |   |   |   \-- ✅ SignalsList.tsx
|   |   |   |-- ✅ AnalyticsCards.tsx
|   |   |   |-- ✅ ApiTest.tsx
|   |   |   |-- ✅ BalanceCard.tsx
|   |   |   |-- ✅ CandlesChart.tsx
|   |   |   |-- ✅ Chart.tsx
|   |   |   |-- ✅ ChartView.tsx
|   |   |   |-- ✅ EquityChart.tsx
|   |   |   |-- ✅ ErrorBanner.tsx
|   |   |   |-- ✅ Header.tsx
|   |   |   |-- ✅ LoaderOverlay.tsx
|   |   |   |-- ✅ PriceChart.tsx
|   |   |   |-- ✅ RiskCards.tsx
|   |   |   |-- ✅ RiskMonitor.tsx
|   |   |   |-- ✅ Sidebar.tsx
|   |   |   |-- ✅ SignalDetail.tsx
|   |   |   |-- ✅ SignalFeed.tsx
|   |   |   |-- ✅ StatsCard.tsx
|   |   |   |-- ✅ StatusCard.tsx
|   |   |   |-- ✅ StressTrendsCard.tsx
|   |   |   |-- ✅ Toast.tsx
|   |   |   |-- ✅ TradeLogs.tsx
|   |   |   |-- ✅ TradeTable.tsx
|   |   |   \-- ✅ Watchlist.tsx
|   |   |-- ✅ hooks/
|   |   |   |-- ✅ useAutoRefresh.ts
|   |   |   |-- ✅ useCandlesPoll.ts
|   |   |   |-- ✅ useCandlesWS.ts
|   |   |   |-- ✅ useDarkMode.ts
|   |   |   \-- ✅ useDashboardData.tsx
|   |   |-- ✅ pages/
|   |   |   |-- ✅ Backtest.tsx
|   |   |   |-- ✅ DashboardPage.tsx
|   |   |   |-- ✅ Settings.tsx
|   |   |   \-- ✅ Trades.tsx
|   |   |-- ✅ services/
|   |   |   |-- ✅ api.ts
|   |   |   \-- ✅ twitterService.ts
|   |   |-- ✅ types/
|   |   |   \-- ✅ index.ts
|   |   |-- ✅ utils/
|   |   |   |-- ✅ api.ts
|   |   |   |-- ✅ position.test.ts
|   |   |   |-- ✅ position.ts
|   |   |   \-- ✅ ws.ts
|   |   |-- ✅ App.tsx
|   |   |-- ✅ counter.ts
|   |   |-- ✅ index.css
|   |   |-- ✅ index.tsx
|   |   |-- ✅ main.ts
|   |   |-- ✅ main.tsx
|   |   |-- ✅ setupTests.ts
|   |   |-- ✅ style.css
|   |   |-- ✅ typescript.svg
|   |   \-- ✅ vite-env.d.ts
|   |-- ✅ .env.example
|   |-- ✅ .gitignore
|   |-- ✅ Dockerfile
|   |-- ✅ Dockerfile.test
|   |-- ✅ index.html
|   |-- ✅ nginx.conf
|   |-- ✅ package-lock.json
|   |-- ✅ package.json
|   |-- ✅ postcss.config.tsx
|   |-- ✅ PR_WAVE1_BODY.md
|   |-- ✅ PR_WAVE2_BODY.md
|   |-- ✅ PR_WAVE4_BODY.md
|   |-- ✅ README.md
|   |-- ✅ tailwind.config.tsx
|   |-- ✅ tracked_files.txt
|   |-- ✅ tsconfig.json
|   |-- ✅ vite.config.ts
|   |-- ✅ vite.config.tsx
|   \-- ✅ vitest.config.ts
|-- ✅ run-18053570799-logs/
|-- ✅ scripts/
|   |-- ✅ stress/
|   |   |-- ✅ tests/
|   |   |   |-- ✅ test_experiments.py
|   |   |   |-- ✅ test_generate_report.py
|   |   |   |-- ✅ test_retention.py
|   |   |   \-- ✅ test_run_frontend_repeats.py
|   |   |-- ✅ experiments.py
|   |   |-- ✅ generate_report.py
|   |   |-- ✅ harness.py
|   |   |-- ✅ merge_frontend_into_aggregated.py
|   |   |-- ✅ rebuild_aggregated.py
|   |   |-- ✅ retention.py
|   |   |-- ✅ run_frontend_repeats.py
|   |   |-- ✅ summarize_frontend.py
|   |   \-- ✅ upload_artifacts.py
|   |-- ✅ tests/
|   |   \-- ✅ test-credread.ps1
|   |-- ✅ tools/
|   |   |-- ✅ npm_audit_triage.py
|   |   |-- ✅ nssm.exe
|   |   |-- ✅ openai_agent.py
|   |   \-- ✅ openai_code_example.py
|   |-- ✅ _fetch_failed_runs_logs.ps1
|   |-- ✅ _helpers.ps1
|   |-- ✅ auto_fix_mypy.py
|   |-- ✅ ci-runs-summary.ps1
|   |-- ✅ ci-watch.ps1
|   |-- ✅ create-audit-fix-pr.ps1
|   |-- ✅ create_service.ps1
|   |-- ✅ create_startup.ps1
|   |-- ✅ download_and_archive_runs.ps1
|   |-- ✅ download_failed_run_logs.ps1
|   |-- ✅ download_run_logs.py
|   |-- ✅ download_runs.py
|   |-- ✅ generate-reexports.js
|   |-- ✅ generate-reexports.ts
|   |-- ✅ head_bytes.ps1
|   |-- ✅ install-dev.ps1
|   |-- ✅ install-dev.sh
|   |-- ✅ install_nssm_and_configure_service.ps1
|   |-- ✅ install_nssm_service.ps1
|   |-- ✅ list_zip_entries.ps1
|   |-- ✅ local_ci.ps1
|   |-- ✅ monitor_pr_ci.py
|   |-- ✅ poll_runs.ps1
|   |-- ✅ README-ci-watch.md
|   |-- ✅ README-GH-Auth.md
|   |-- ✅ README.md
|   |-- ✅ recover_archived_runs.ps1
|   |-- ✅ register-ci-watch-task.ps1
|   |-- ✅ repair-dev-deps.ps1
|   |-- ✅ repair-dev-deps.sh
|   |-- ✅ scan_failed_logs.ps1
|   |-- ✅ service_wrapper.cmd
|   |-- ✅ setup-dev.ps1
|   \-- ✅ setup-dev.sh
|-- ✅ tests/
|   \-- ✅ _helpers/
|       |-- ✅ external_data_stub.py
|       \-- ✅ train_and_save_stub.py
|-- ✅ tools/
|   \-- ✅ predict_sample.py
|-- ✅ .bandit
|-- ✅ .env.example
|-- ✅ .gitattributes
|-- ✅ .gitignore
|-- ✅ .pre-commit-config.yaml
|-- ✅ .secrets.baseline
|-- ✅ _regen_tree.py
|-- ✅ ARCHITECTURE.md
|-- ✅ CHANGELOG.md
|-- ✅ check-containers.ps1
|-- ✅ CONTRIBUTING.md
|-- ✅ DEVELOPMENT.md
|-- ✅ docker-compose.yml
|-- ✅ fix-problems.ps1
|-- ✅ git-push-all.ps1
|-- ✅ LOCAL_CI_README.md
|-- ✅ main_train_and_backtest.py
|-- ✅ MIGRATION_PLAN.md
|-- ✅ mypy.ini
|-- ✅ PROJECT_STATUS_TREE.md
|-- ✅ PULL_REQUEST_DRAFT.md
|-- ✅ PULL_REQUEST_WIP_GHCR_USE_GITHUB_TOKEN.md
|-- ✅ query
|-- ✅ Read.md
|-- ✅ readme.md
|-- ✅ requirements-ci-upload.txt
|-- ✅ requirements.txt
|-- ✅ run18066333212_jobs.json
|-- ✅ TEAM_NOTIFICATION.md
|-- ✅ tmp_jobs_18066605412.json
\-- ✅ TODO.md
