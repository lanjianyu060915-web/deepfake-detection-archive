# 非 Git 等价差异报告

- 基线：`D:\刑侦查竞赛\face_source_trace_two_search_v0.1_official`
- 目标：`D:\刑侦查竞赛\face_source_trace_three_module_v0.2_dev`
- 算法：相对路径集合 + 文件 SHA-256；文本文件附 unified diff。
- 排除：任意层级 `__pycache__`/`.pytest_cache`，以及 `runtime/tasks/`、`runtime/inbox/`、`runtime/validation/`、`packaging/`、`release_v1.0/`、`docs/EQUIVALENT_DIFF.md`。
  排除项是运行时任务/缓存噪音和继承的 v0.1 大型打包产物；正式测试证据另归档在 `test_results/`。

## 汇总

- 新增：112 个文件
- 修改：18 个文件
- 删除：0 个文件

## 新增文件

- `RELEASE_MANIFEST.md` — 2,401 bytes — `89bbaa346b79d0396480e3b7a3875e03f6e5e3c94e4abe2b1e9b1cc9aeffc83b`
- `docs/AIGC_SOURCE_LOCATION_AUDIT.md` — 5,004 bytes — `f91161bcc4b2f7c4840d556aa1edd97ac21af876bf7d1ad9f2a32e265c633084`
- `docs/BASELINE_COPY_RECORD.md` — 1,411 bytes — `b928ecd71e5e26faa9b42d69c150d25de19624e8330943660aafeaefda2fa188`
- `docs/CHATGPT_REVIEW_PACKAGE_README.md` — 1,132 bytes — `7db7d7e6d2ea699799c9fec38f373769f167c84c5aee26ef974be102fcbe0a86`
- `docs/CHATGPT_REVIEW_STATUS.md` — 2,253 bytes — `a98fd0ecea031df36c167b151e2ac25c61756faddacdbef656cfded5dba23da8`
- `docs/DATABASE_AUDIT.md` — 3,007 bytes — `4a1bcd171d86b8bb1bdbaaf28cf60ac447568e1f5a524f3060dc9d564c161dda`
- `docs/DEPENDENCY_AUDIT.md` — 2,200 bytes — `78e2df215cad624acb7e6d8669ac34dd511aadddb3c793a3af5196e60d7a3e56`
- `docs/MODEL_AND_RUNTIME.md` — 4,073 bytes — `49634cccd483ecfda4af965dc02926a3c4b00007d4f2e8a66ce4e41ff8c12a2f`
- `docs/MODIFIED_FILES.md` — 3,698 bytes — `50f1eb3d820957dd2cc6aa72b624edb2971c2a0e49daa7f9e4a43acf9fc06156`
- `docs/REGRESSION_TEST_PLAN.md` — 3,117 bytes — `1c64244f14bac6d1862e9ae8a7ca727f1be344c133cd47472b06f6be0f1e699d`
- `docs/REGRESSION_TEST_REPORT.md` — 7,399 bytes — `8187849e756b20d92b989385b9c0316e10fed035817125f32be6fdda72126bad`
- `docs/RUN_STATUS.md` — 1,535 bytes — `3a30eecf4ff9d3ac963ccb2b5775e50b5bceaeb50b972d0a6ba158f12bc20c2d`
- `docs/THREE_MODULE_ARCHITECTURE.md` — 2,062 bytes — `894bf252dae4a9dec35367fb8ec2ef99a750b8d32349b426eb08031748363551`
- `docs/THREE_MODULE_GUIDE.md` — 2,044 bytes — `856beaf6fb0b3c4f551fb5c3cdebd8ac85e49620fd871151e212895d24088cc7`
- `examples/README.md` — 636 bytes — `a1c6d0e9b7d287de256fd06484710115967c9cd91ca247677de2469a1506797e`
- `examples/aigc_fake_image.jpg` — 2,713 bytes — `796ba61010dc071ea19e8f0b2877c68beae085a26e4d2e4b26acf4317b5da9f7`
- `examples/aigc_real_image.jpg` — 3,394 bytes — `6d5774ecbea891f1d1d7e8c652b9c1cddccc4330117dc4b429067ea653da1418`
- `examples/integrated_query_001.jpg` — 295,848 bytes — `8b1c6f43491d0f2e179d0950ff285f67001c9fce90eb9e0fc12facbc63249dd6`
- `examples/integrated_query_002.jpg` — 52,719 bytes — `fc9999f381a48b9100ce6a6591923e83ed07d8ec21fd8ba0f8d81b7ec3d06079`
- `examples/integrated_query_003.jpg` — 36,905 bytes — `e7d701f05d0ffadd8630df49bcfa3dd749f6d0e3a4d0aff862f23335b85f44b0`
- `examples/no_face_image.jpg` — 13,645 bytes — `4d64895ea9a89816111eece5501d21cefe3ff865e64274b45373a426caff6dc2`
- `frontend/FRONTEND_INTEGRATION_REPORT.md` — 3,393 bytes — `c174124b0b3539a8e0db3e090d047bd95bd51d3016dbf0a6f851103c0505eb20`
- `frontend/components/status_adapter.py` — 6,860 bytes — `a9d318e281118443be36094221d1f32f8e379841e776056e1203e0261f0f4711`
- `frontend/mock_results/detection_result.json` — 454 bytes — `b0d6b875d6b611da221dbbeff57a853e585d1a70a768dbd6f2307e5e9d078a50`
- `frontend/mock_task/detection_result.json` — 441 bytes — `ab356855887119bb6f8f9dcbc010a832ec19c8cc1d4c961d13879402bdc3b6fe`
- `frontend/tests/disabled_services_config.json` — 934 bytes — `a197542ed1b8a262e4ada1bb05353be75aeace7cdfc2896a09419fac79cbd278`
- `frontend/tests/test_three_module_frontend.py` — 8,357 bytes — `554f202719876be5bd48b14ea9b5fc6679bd4600f8dd9e22465de4bff78f9d09`
- `modules/deepfake_detection/README.md` — 1,756 bytes — `d84defc5df1ab22e53ee69cfd8679a0cd32514ae1ac13a6c9220402c0c6242fa`
- `modules/deepfake_detection/inspect_installed_archive.py` — 880 bytes — `f2b46b0102afcca041f8a173ac9e9279548483632ba4932f25a4086eede03dba`
- `modules/deepfake_detection/persistent_worker.py` — 8,912 bytes — `a0ebcff09909a991478fb489762f07367ac1395b21b64162ba43e7fc6ff8c356`
- `modules/deepfake_detection/pyinstaller_pyz_runtime.py` — 9,988 bytes — `c6f3836d4c9252cbb2fc315acfd97e78b5f0b8bf10941e607bea996bcb86d836`
- `modules/deepfake_detection/run_deepfake_detection.py` — 9,949 bytes — `fd74aaa13996f735f9bbbe86afd6d2919c3bee5c0722c019c9f3792be6aeab25`
- `modules/identity_search/persistent_worker.py` — 9,734 bytes — `90454c6a9f5fba7fa41693b3661836d10a79fafc6f6e6f9cc2bac4e322e082b8`
- `requirements-aigc-frozen.txt` — 379 bytes — `f6a67b97fe20f32608d415488b8e4ffb6914a244efa98c65ba333c9e6d04033d`
- `requirements.txt` — 340 bytes — `4db9680c1f867789d7dda96f9117953a467ce263277f8e6d225a83d3f495ace7`
- `runtime/cache/deepfake/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth` — 10,306,551 bytes — `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`
- `runtime_services.py` — 37,621 bytes — `063dd63bd45952e404f47e6a97fa14fcfdc06c361645a631f05c43759f975f77`
- `screenshots/three_module_home_ready.png` — 44,695 bytes — `1babf1451826fc6366a5d610934070bc91c2bbe19b72ec4b6ff4a2cd1f4a6ecf`
- `screenshots/three_module_identity_results.png` — 70,704 bytes — `bad427ad0af85106cec7a7903fa1c87d6e26f45a2cb1fda4183934b818b1a476`
- `screenshots/three_module_live_progress.png` — 40,820 bytes — `335a460203b0ff3dd2b803df9d2cf9cb162fbc6cc19426063e158fbfda9a187e`
- `screenshots/three_module_live_result_summary.png` — 72,926 bytes — `5943eb52717c58f66d296a3eef8075bb60ee139fe7e33837ff5221c1bef53b4b`
- `screenshots/three_module_source_results.png` — 74,864 bytes — `2c558c7845530267224550fb53617534e32ec7ae4c740895fb59d3536d948f50`
- `screenshots/three_module_upload_four_actions.png` — 36,903 bytes — `6ce61ed674319b94a673b1da06dc18cb187315b524ad4a6806ba0594d327a9b3`
- `start.bat` — 188 bytes — `52a84c40d1bfee1f9d26a0055102e451d7990eb57c5d10f5c2768fd0ab65f522`
- `start.ps1` — 3,044 bytes — `9b0d5a740a6f3ae0e3cb81157c909ddad02b42c10a324d5aa14acb810404f235`
- `system_status.py` — 33,674 bytes — `c839be38ddd397323d9978034dffb2cb08d927f367303e5805423a13c236e8dc`
- `test_results/README.md` — 1,068 bytes — `6ece6f56686079e13b1bb334443eb476d82031d63efc292ef7cae23a0a22ab77`
- `test_results/combined_query001/detection_result.json` — 2,529 bytes — `6b199e420205ac20ffddcaa401d1fcb51d0d25b86827d0750a6ba61148ed29ed`
- `test_results/combined_query001/identity_result.json` — 1,497 bytes — `07a5e19f6ccc79ee6f9215726b4043c9ab8c67b89327d3d8544897abdd4c6e3c`
- `test_results/combined_query001/source_result.json` — 2,293 bytes — `c9a6dcb30a74d70b906c3ecde06ec89fafcf7ff715acb999fa5e4d48ed77f926`
- `test_results/combined_query001/task_status.json` — 3,377 bytes — `0875d5a5b5ee7ddb3bdafc535cca027bb21d636ca55689581c0331b33c51d6b3`
- `test_results/combined_query001/unified_result.json` — 760 bytes — `0927b7bcf04d1d09d14ef9e5294012c781c7080cf1e6f85a7cbd17c697a4843d`
- `test_results/combined_query003/detection_result.json` — 2,527 bytes — `7b2cb718b6028ef5ade8010ff266ee2acf607e42821b26be3cfc9ba196cad2a5`
- `test_results/combined_query003/identity_result.json` — 1,497 bytes — `882181ade4e082e63dd4a895cb143b640d55a50ae92fcd3dd786893302fd5e0f`
- `test_results/combined_query003/source_result.json` — 2,281 bytes — `edb45de09606312e843eb8da3d251a3e2c4f8bb7850d6a5b03b90cea58cb384d`
- `test_results/combined_query003/task_status.json` — 3,381 bytes — `8634834ebcb44b7f8755835666ac6b703d8ffc25ec2267b54a4e3c9138ba7163`
- `test_results/combined_query003/unified_result.json` — 764 bytes — `a79c2ba36f65926f9674d026832810d4254a83b99e70e24564ee51b8313dc271`
- `test_results/direct_module_results/aigc_fake.json` — 2,480 bytes — `10b30bf667a2b324f7df409ea5a038a1692835e3ad47910693ed08114ef0b67c`
- `test_results/direct_module_results/aigc_invalid_format.json` — 2,611 bytes — `bd1844e2963a5acc97a2b7dbedb5a565c67316cac517d2d6de1aeaa7fbc741e4`
- `test_results/direct_module_results/aigc_missing_path.json` — 439 bytes — `7041fd48c34414251ac3ffeb2cc9a3e8e5996a24032b8f21ada6ffa0faa0aa38`
- `test_results/direct_module_results/aigc_no_face.json` — 2,527 bytes — `8754d0ef74a1292cbea472c1e9e80eaaf77d1e1402bdb3290ec9870f771ac5b2`
- `test_results/direct_module_results/aigc_real.json` — 2,479 bytes — `66e25fe1bcf40b48a675836a95f1d8ccd195ad8dd231db763bf6c23c5693297f`
- `test_results/direct_module_results/identity_empty_library.json` — 401 bytes — `d632cb705e79251aec0e1931c16480d1c08c294e639d7dc143287c066ea5ec64`
- `test_results/direct_module_results/identity_multi_face.json` — 400 bytes — `39a62ce80d10b32c7eee3bd5c28b2dcf99ad033cb7d2e5cebb7725716ef4fed8`
- `test_results/direct_module_results/identity_no_face.json` — 335 bytes — `a6be63a93663fe98ed90750eb4cfcbc5ef40bb51e82c2ecc63274d0227c10c51`
- `test_results/direct_module_results/identity_out_of_library_external.json` — 1,459 bytes — `72a608750e107bd8aee2950328504aaf4c2270a7dd801723cedd96e0ba2edee7`
- `test_results/direct_module_results/identity_query_001.json` — 1,435 bytes — `7ce44ebcc4e9d561dea8fd456e57c6dd94ef4e307e8be05549a6275a31164e5e`
- `test_results/direct_module_results/source_compressed_122.json` — 2,217 bytes — `c6007c27d38781558a74a40b8cf4df010da37351146a4632c000db8da20ddb9c`
- `test_results/direct_module_results/source_compressed_122.runtime.log` — 426 bytes — `acdae72c975708f575f7e21392494f32b325f0d0e334ea34ba22e81af840dd6f`
- `test_results/direct_module_results/source_cropped_122.json` — 2,208 bytes — `9af4e97424af8ee52051e8ea2a443e4cd3e98fb86fe7280b326cdf18642f6fc2`
- `test_results/direct_module_results/source_cropped_122.runtime.log` — 429 bytes — `98b77ad6c120fd4d129bb189463fbe4eb62b8036c7e36f4dab0fb0daf14c7dfd`
- `test_results/direct_module_results/source_empty_library.json` — 266 bytes — `d33c8afd53df405f1bf408bff4f6330ffed83ef50446b190058447249345b5c1`
- `test_results/direct_module_results/source_empty_library.runtime.log` — 1,118 bytes — `3f847ab5f2a00cbb1e0085118be9df37325eafa847ff033ac574c752951d439c`
- `test_results/direct_module_results/source_original_122.json` — 2,142 bytes — `0948ab7b51c8ab374b414a6526474d0f869665f7f5f777dcbc7c6b1bf42ad3f1`
- `test_results/direct_module_results/source_original_122.runtime.log` — 375 bytes — `6868586610750942975c85d32323d9133b3b5cc2580807d9188550881c7dc772`
- `test_results/direct_module_results/source_out_of_library_external.json` — 2,260 bytes — `528fb9a2d33a114d57a8f5fcdc2a6fd538e3e2751c9983a5961c776160c0ba57`
- `test_results/direct_module_results/source_out_of_library_external.runtime.log` — 422 bytes — `a73f7101e8180e4457ede362cb076d1c0c280af45aa8ab0b97f91fca2c228e01`
- `test_results/environment_status_summary.json` — 1,217 bytes — `8aca6fa40543561b6c1c007fe19d69c9eb317de80184ab37a645e490f7dbc22e`
- `test_results/failure_isolation/detection_result.json` — 2,529 bytes — `5f6eb4e1e8bc902b5bc686546dbe5174bda2231978d6368e191cbaf95b788711`
- `test_results/failure_isolation/identity_result.json` — 396 bytes — `4a1693a97dc1e9b53891cd07ec75bd67717d143c8b8fd4a96902898bae1183d7`
- `test_results/failure_isolation/progress.json` — 1,145 bytes — `ace454732afc89b74627f9b3a6ed07515e2c1cec2c1b2490dd01a59de02d58fe`
- `test_results/failure_isolation/source_result.json` — 303 bytes — `e1e976f622893340a1489b63955ce62fe5d237370eeb6fe4fb54cc339d04eda2`
- `test_results/failure_isolation/task_status.json` — 3,687 bytes — `9063840e5f43084e3bb46a37ca9d92e590ea9d85f1d967a72799d522dc75cf43`
- `test_results/failure_isolation/unified_result.json` — 928 bytes — `ee53c3e19b2aa3b2d9d33e747e9ed380026df5fe676c85f34b7cd7cae6991065`
- `test_results/persistent_service_summary.json` — 10,274 bytes — `f891efafe1ee09322f73a5bf979a4e89ce5e8b2a0abc02ad0196f29ef0fedea8`
- `test_results/release_real_smoke.json` — 10,546 bytes — `d484969b7ef2fc792aaa260a1ac33974047a633af7271b44504b5fb2047aaa2f`
- `test_results/release_smoke/detection_result.json` — 2,574 bytes — `5291218e1908d106e91051a0a727184a3c27d09fcecde936665a3572c8e87ded`
- `test_results/release_smoke/identity_result.json` — 1,512 bytes — `e0778cac863f20464733a8deb64b396b36997f6a12f5fe120b69eae8d1f68ed1`
- `test_results/release_smoke/source_result.json` — 2,382 bytes — `869e9a8e8fbc94a5c3b08b2ca181ec744063663909d28a17531180174c04779b`
- `test_results/release_smoke/source_result.runtime.log` — 439 bytes — `56b782063e2b51876aa0e18e3ef0976e2a06409278173b366413f2fab3d88c8b`
- `test_results/release_smoke/task_status.json` — 3,575 bytes — `eb52a66c19fa7100e596019be58cc390350c501383b2146274ca510bf6779aa5`
- `test_results/release_smoke/unified_result.json` — 763 bytes — `3b1a46bb1799e4494f000899f5c63175e1dd7d974c4be669daea279e934fc6a9`
- `test_results/ui/three_module_home_ready.png` — 44,695 bytes — `1babf1451826fc6366a5d610934070bc91c2bbe19b72ec4b6ff4a2cd1f4a6ecf`
- `test_results/ui/three_module_identity_results.png` — 70,704 bytes — `bad427ad0af85106cec7a7903fa1c87d6e26f45a2cb1fda4183934b818b1a476`
- `test_results/ui/three_module_live_progress.png` — 40,820 bytes — `335a460203b0ff3dd2b803df9d2cf9cb162fbc6cc19426063e158fbfda9a187e`
- `test_results/ui/three_module_live_result_summary.png` — 72,926 bytes — `5943eb52717c58f66d296a3eef8075bb60ee139fe7e33837ff5221c1bef53b4b`
- `test_results/ui/three_module_source_results.png` — 74,864 bytes — `2c558c7845530267224550fb53617534e32ec7ae4c740895fb59d3536d948f50`
- `test_results/ui/three_module_upload_four_actions.png` — 36,903 bytes — `6ce61ed674319b94a673b1da06dc18cb187315b524ad4a6806ba0594d327a9b3`
- `test_results/ui_live_task/detection_result.json` — 2,529 bytes — `f9aea1fa678f1ebf94d185952a51fb58d401227da3a1c1434845921cd754f2ca`
- `test_results/ui_live_task/identity_result.json` — 1,497 bytes — `ee63ea46089edb1d039ed32160b463a9a75fe9ee92ceb190d140abbeb1e195fe`
- `test_results/ui_live_task/progress.json` — 2,006 bytes — `4396c148ed1697d27d163313dee47f3318a610dfcc8ab015033413419cd99c8e`
- `test_results/ui_live_task/source_result.json` — 2,293 bytes — `2799caf4bbdc0633970de2c1d9479ecf5051588a64b5d06112af7153e8680e07`
- `test_results/ui_live_task/task_status.json` — 3,389 bytes — `18f7d43f9d8ab90089b23103c0f8e95015674c22dd4d56ee675f64771e4ad49b`
- `test_results/ui_live_task/unified_result.json` — 763 bytes — `96c0c08216221383e27710b879da070951f65d450570cc6f93010ef0f9dc0d08`
- `test_results/unit_test_log.txt` — 736 bytes — `a53b57fe1ea23f7eba71b6536e1e00ecd949dbc72daca3450c6f08e052be8ac8`
- `tests/fixtures/mock_persistent_worker.py` — 2,492 bytes — `109d650f1404ea28893db5588fedb4795034e013245940ffb7e8d821adcd5283`
- `tests/fixtures/not_an_image.txt` — 153 bytes — `1d7519ab7bb6c0a9efcb996eed439ad4dfc12017448c9cf1a31fc36991c23ecb`
- `tests/fixtures/slow_search_runner.py` — 234 bytes — `687a48dc925617ed47a9a21258bae818b4cf33b10d29a001aee7749b7e71a742`
- `tests/run_real_integration.py` — 4,019 bytes — `5b7432cbb746068885dc8180f1bd0e13d6ed841ac585e880678997ff0aa8bafd`
- `tests/test_runtime_services.py` — 6,282 bytes — `9731ca4486fb4ef7fa05839e69b02c22e13f061809c1d0417deeae7ad4faa6d9`
- `tests/test_three_module_backend.py` — 14,189 bytes — `ce5b2fed326bdc5835df90c2f773b0ebcb995e01e63f079d0e474e47e83ec90e`
- `tools/generate_equivalent_diff.py` — 6,285 bytes — `f1eeedc239189eaf7fc96b0d1e04159556a8f64fff1f9ebf61180646564d4d77`

## 修改文件

- `CHANGELOG.md` — 621 → 1,822 bytes — `032fcfb01abc58e560dfc801f440fc6f6dc6454c213a079ed02478f7bb407e9b` → `b25de4ec5670787e2b46d5452047a47a29a0dc2b309b7c4c8dec14b3059a87bd`
- `KNOWN_ISSUES.md` — 844 → 2,138 bytes — `8825e51e5383b8f0f67d20e88b9f03085bd25b2ef9252568e8c63315a4df1da1` → `eebe98a9d788a4fbc4e4441a4b99c269573515bd982dbcf1777dae0988871047`
- `README.md` — 4,096 → 6,604 bytes — `577edd58b0e1a94252e89d273596f05f42ce7828c8d7735bbe2a7c157840da96` → `8a7840aaa5d329bebf87650abd57c1090dbe1cd381190de53011516943ebb907`
- `check_environment.py` — 8,672 → 9,684 bytes — `1b64dcfe2acde3f4840bad73f88e7355809106c7e6b09de549fbeb6c54dc7602` → `27b11bf6eb2efb3fc03543a3b70f831c7a7a0abd59c41cc2a15a857df4b56930`
- `config.json` — 935 → 2,553 bytes — `fede924962f89ef748c82f506e33200d8429c7a63cbc3691ab11fa97ca2e940b` → `764c7a9fb6dd1d0b5d77dc5bdd580ec182e4cf6141a0196ddfefa970665f7ad5`
- `docs/FRONTEND_VERIFICATION.md` — 805 → 1,477 bytes — `9f2bbe6e581875df1d58e831ab4591340c6dc8715855d988b81ef198f8b1cb91` → `fb242acb7808a93077165219a4d1578d9ccb60956e968b733d3914a21fde9917`
- `frontend/README.md` — 3,270 → 2,635 bytes — `bc6414d5da9ea41ed0a573e98b35738278b956afe9ddafe3556c1d8b721c34b5` → `e4e7ed676b147ec853cad4ca276c01178dd43b97958010a4067340c84920dba5`
- `frontend/TASK_DIRECTORY_CONTRACT.md` — 1,806 → 2,403 bytes — `d428db6382983492effbea7c8ef915ed359f756d3c5e4313a6c112bb509e93dc` → `e68f847d36df5499c9e6d26f39e87646c4d8e3349199fb086a17f46828e0d602`
- `frontend/app.py` — 13,838 → 30,017 bytes — `e46c91aa6dcb228acbe99127f15f1985d3b360d8a30c14e4d017dfd2cd81a07b` → `a1445fbe1549d1243829caae794aa7ef317469e0ee753dd7b5a78a6dd97aba0e`
- `frontend/components/data_loader.py` — 8,825 → 13,280 bytes — `09d263ace873e7e517829b6076e16571d6bed11f1e31c5a8ab728916337df706` → `a13a6fea67d7f3da59a36153ec4ccae842b065c553fb654035b2ab0f08dbf549`
- `frontend/components/result_views.py` — 5,237 → 13,708 bytes — `c704ede8a7f74392102d85364458583de684ac707fa7c0ebd5f661ae265a9d21` → `0aa06f8ec8cfaea6a00e96f348e4205e3ba706d2a19f66bc0e01c5d11b689864`
- `frontend/mock_task/README.md` — 238 → 266 bytes — `042f13383ea594501ae02dca053c74c0da4117a0c431961a115ed7f1e058949e` → `a4a492ca652379af536887d8dd715868c4604ec9c28c9d610a581160a0ef100f`
- `launcher.py` — 1,836 → 2,325 bytes — `e067a9af200111ad23ebffdec4ad35000f7b1773fd349c0d7258f3aa814fa80a` → `7d1f121b0adefff1076dba002125924f64e9da61284b3d67137571c4bfc2423f`
- `orchestrator.py` — 17,166 → 48,907 bytes — `730fd7522ea904e18d1222e39b69cbc6a8317295dd603167a95c01bacd241685` → `a3339fa2c976e2341de4a693c3cc764dd86b84d6c0ada8b97f1eafa1a4ece0bd`
- `run_app.bat` — 527 → 79 bytes — `9695fedf838bf0d60aa5decf09e5b5d754fa2a8181015195e1d5d353f48b9f79` → `a49bcfbe2038662a618bdc7f712e3def74ab95a1019b78fffca594d74e285e55`
- `tests/README.md` — 423 → 761 bytes — `2ceb969862b248f289b9462205d1c921e4b50c04d682a369ceec4d6b97dbd08e` → `88ce7678cc61920eeea45855e36952b2f5ce5400a34459b0989b566b4bb85615`
- `tests/fixtures/README.md` — 393 → 555 bytes — `7422017fa9a2c74886dfa62ec54d54ef06e5312ee23b4885105fb7531f1baed9` → `1d5d4144b2e18fb8c8ccfaaa240c07166a62509cd2789a913bdef8d4729127cc`
- `tests/test_orchestrator_contract.py` — 4,386 → 4,721 bytes — `adb552e751b728e74095d44cf8e6a26c248473b9e5203181a90c3145b8a30b57` → `1899e2962b95740bddd281292e1325941b1e8a81e1fafe783ef4f580c864ac07`

## 删除文件

- 无

## 文本 unified diff

### `CHANGELOG.md`

```diff
--- base/CHANGELOG.md
+++ target/CHANGELOG.md
@@ -1,12 +1,22 @@
-# CHANGELOG
+# 修改日志
+
+## v0.2-three-module-official — 2026-08-25
+
+- 从 `face_source_trace_two_search_v0.1_official` 完整复制创建独立 dev，原 official 保持只读不变。
+- 审计 AIGC 资料、验收包与已安装程序，确认可验证源码形态为安装目录中的可读模块/打包归档；历史冻结适配器只读加载真实安装版算法和权重。
+- 新增 AIGC 冻结适配器、常驻 worker、统一 `detection_result.json` 契约与模型哈希检查。
+- 为 AIGC 和 InsightFace 新增独立持久运行服务，模型跨查询复用；保留真实子进程 fallback、超时和进程清理。
+- 扩展 orchestrator 为 `deepfake/identity/source` 三分支及 `selected_modules`，实现 skipped、partial_success 和全模块故障隔离。
+- 将底图默认超时从 180 秒调整为 600 秒，匹配 203 张图库真实运行时间；算法、权重和排序未改。
+- 新增 `system_status.py`，报告 AIGC 资源哈希、身份人物/图片/embedding/index、底图图片/特征/index 和实际运行设备。
+- 延续现有 Streamlit UI，增加首页三模块状态、四个操作、进度、错误、三结果页签和无概率融合的证据摘要。
+- 增加 `start.bat/start.ps1`、环境检查、示例、真实集成 runner、模型/依赖/数据库/测试/架构文档。
+- 将外部 A/B 环境和图库配置改为本机绝对路径，修复 dev 复制到 `releases` 后 `../first_week` 解析错误。
+- 完成真实 AIGC、身份、底图边界测试及多案例三模块联合回归；详细数据见测试报告。
 
 ## v0.1-two-search-official — 2026-08-20
 
-- 集成人脸身份搜索 A 与原始底图搜索 B，保持冻结算法和两个独立 Python 环境。
-- 建立统一 orchestrator、launcher、task 目录、状态与日志。
-- 复用第1轮清理后的 C 前端，Live Analysis 真实调用 orchestrator。
-- 身份缓存跨 task 复用；B 匹配图直接归档到本 task。
-- 实现单模块失败不阻断另一模块的故障隔离。
-- 增加环境检查、批处理启动入口和受控异常测试。
-- 通过 query_001、query_002、query_003 三案例真实回归，Rank-1 与冻结结果一致。
+- 正式集成人脸身份搜索 A 与原始底图搜索 B。
+- 建立统一 task、状态、日志、故障隔离与前端结果展示。
+- 通过 query_001、query_002、query_003 双搜索真实回归。
 
```

### `KNOWN_ISSUES.md`

```diff
--- base/KNOWN_ISSUES.md
+++ target/KNOWN_ISSUES.md
@@ -1,22 +1,30 @@
-# 已知问题
+# 已知问题与边界
 
-## A 冷启动
+## AIGC 原始开发仓库缺失
 
-首次模型和身份库特征初始化耗时明显较高。后续 task 会复用 `runtime/cache/identity`，但缓存不随发布包分发。
+当前工作区没有找到带版本控制、训练脚本和构建脚本的原始开发仓库。可验证接入来源是已安装 `DeepfakeDetectionSystem` 的 PyInstaller 归档、可读模块和真实模型文件，并由历史冻结适配器只读加载。安装目录不是本次可重构源码仓库，且未被修改。
 
-## B 速度
+## AIGC/身份当前实际为 CPU
 
-200 张候选库检索仍需要几十秒，尚未做候选固定特征预计算和索引优化。本版本不实施性能优化。
+本机有 NVIDIA GPU，但 AIGC 安装版是 `torch 2.11.0+cpu`；A_env 的 ONNX Runtime 也没有 CUDA provider。系统会如实显示 CPU。若要使用 GPU，必须重新准备并验证对应 CUDA 运行环境，不能只改 UI 标签或 `device` 字符串。
 
-## 环境
+## 身份搜索是闭集候选排序
 
-当前版本依赖本机已经验证的 A/B Python 环境和本地 InsightFace 模型；发布包不携带 venv 或模型，换机后需更新 `config.json`。
+系统没有经过验证的开放集拒识阈值。库外人脸仍会返回相对 Top-K，应结合相似度和人工复核，不得把 Top-1 自动解释为身份确认。默认 45 张参考图中当前 43 张形成有效 embedding；两张未检出可用单人脸，不影响其余缓存就绪状态。
 
-## 功能边界
+## 底图检索没有持久 index
 
-当前仅支持图片输入、闭集身份库和封闭底图库。暂无视频、开放集拒识、互联网搜索、案件管理、报告生成或换脸鉴伪。
+冻结 B 算法对 203 张图库逐图提取特征，单次 CPU 真实检索约需 1.5–4 分钟，受并发、图片尺寸和磁盘影响。当前未添加离线特征库或 FAISS，避免改变冻结算法；默认超时为 600 秒，UI 会显示运行阶段。
 
-## 前端运行方式
+## 外部依赖不随程序重复分发
 
-Streamlit 在单页面会话中同步等待两个子进程，但会显示阶段状态。当前没有多用户任务队列或并发容量保证。
+正式程序引用本机 A/B 环境、图库、InsightFace 模型和已安装 AIGC 系统。复核 ZIP 不含大模型/数据库；完整正式程序也不复制这些外部资源。迁移到另一台机器时需更新 `config.json` 并重新运行状态检查和真实回归。
 
+## 输入与产品边界
+
+统一 UI 当前只接入单张 JPG/JPEG/PNG/WEBP（20 MiB 上限）。没有视频入口、互联网反向搜索、案件管理、多用户队列或司法报告签名功能。AIGC 安装版虽包含视频能力，本次未把它宣传为已集成功能。
+
+## 结果用途
+
+三个模块的分数定义不同，不提供融合概率。系统结果仅用于风险筛查和侦查线索，应结合原始文件、传播链、设备信息和其他证据复核。
+
```

### `README.md`

```diff
--- base/README.md
+++ target/README.md
@@ -1,99 +1,146 @@
-# 换脸寻源双搜索系统 v0.1
+# AI换脸涉案图像智能鉴伪与原始素材溯源系统 v0.2
 
-本版本把冻结的人脸身份搜索 A、原始底图搜索 B 和第1轮清理后的 Streamlit 前端 C 正式集成为一个图片双搜索产品。上传一张图片并确认两个图库后，系统自动创建独立 task，依次运行 A/B，统一展示身份 Top-5、参考照片、底图 Top-5、候选图和匹配可视化，并完整归档结果与日志。
+本版本以 `face_source_trace_two_search_v0.1_official` 为唯一主基线，新增真实 AIGC 鉴伪并保留原有身份搜索和底图搜索。上传一张图片后可分别运行三个模块，或执行“一键综合分析”。每个模块独立记录结果和错误；综合页面只汇总证据，不生成没有实验依据的融合概率。
 
-本版本不包含换脸鉴伪、视频、开放集拒识、案件管理或报告生成。
+## 当前真实运行状态
 
-## 目录
+- AIGC：**真实模型可用**。只读加载本机已安装 `DeepfakeDetectionSystem` 的 PyInstaller 归档、空间模型、频域模型与融合配置；已知 SHA-256 全匹配。
+- 身份搜索：**真实数据库可用**。默认库 9 人、45 张参考图，当前有效缓存 43 个 embedding，index 状态 `ready`。
+- 底图搜索：**真实数据库可用**。默认库 203 张图；冻结算法为逐图实时扫描，当前没有持久化特征/index。
+- 实际设备：当前 AIGC 安装运行时为 `torch 2.11.0+cpu`，InsightFace 环境只有 CPU provider，因此本机实际推理为 CPU。系统检测到 NVIDIA GPU，但不会虚报为已使用 CUDA。
+
+这些状态会在每次启动时重新检查，而不是写死在 UI 中。
+
+## 一键启动
+
+双击根目录的 `start.bat`。启动流程为：
 
 ```text
-launcher.py                 命令行入口
-orchestrator.py             task、独立子进程、日志与统一状态
-config.json                 本机环境、默认图库、超时和缓存
-check_environment.py        快速环境检查
-modules/identity_search/    冻结 A 运行源码
-modules/source_search/      冻结 B 运行源码
-frontend/                   第1轮 C 与真实 Live Analysis
-runtime/tasks/              每次分析的独立归档
-runtime/cache/identity/     跨 task 复用的 A 身份特征缓存
-runtime/inbox/              前端临时上传区
-tests/                      快速工程与故障隔离测试
-docs/                       集成报告
+环境检测 → 模型/哈希检查 → 数据库计数与 index 检查 → 启动 Streamlit → 打开浏览器
 ```
 
-产品不包含 A_env、B_env、InsightFace 模型、完整图库或验收集；这些由 `config.json` 引用。
+默认地址为 `http://127.0.0.1:8767`（避开本机已有双搜索程序占用的 8766）。启动终端应保持打开；关闭系统可使用 UI 侧栏按钮或关闭该终端。
 
-## 环境
+前端 Python 按以下顺序选择：
 
-- A Python：`first_week\_round1_work\backend\envs\A_env\Scripts\python.exe`
-- B Python：`first_week\_round1_work\backend\envs\B_env\Scripts\python.exe`
-- 前端 Python：优先 `D:\python_3.14\python.exe`，需已安装 Streamlit
-- InsightFace 模型：配置的 `identity.model_root\models\buffalo_l`
+1. `packaging\.venv_frontend\Scripts\python.exe`；
+2. `D:\python_3.14\python.exe`；
+3. PATH 中可导入 Streamlit 的 Python。
 
-A/B 必须继续使用两个独立环境。这样可隔离依赖、模型初始化和失败状态，同时保持已经验收的 CLI 行为；orchestrator 不包含任何算法实现。
+## 四个操作
 
-## 先检查环境
+- `AI换脸鉴伪`：只运行 AIGC。
+- `人脸身份搜索`：只运行 InsightFace 闭集检索。
+- `原始底图搜索`：只运行全库粗排和局部几何复核。
+- `一键综合分析`：依次执行 AIGC、身份、底图，并以三个独立证据块展示。
 
-在产品目录运行：
+未选模块写为 `skipped`。一个已选模块失败时，其余模块继续；任务最终状态为 `success`、`partial_success` 或 `error`。页面会显示加载模型、图像检测、人脸检测、特征提取、数据库搜索和完成进度，并将可理解错误写入任务归档。
+
+## 架构
+
+```text
+Streamlit UI
+  └─ orchestrator（task、超时、故障隔离、统一 JSON）
+       ├─ AIGC Python 3.12 常驻进程 / 真实子进程 fallback
+       ├─ Identity A_env 常驻进程 / 真实子进程 fallback
+       └─ Source B_env 一次性进程（600 秒超时）
+```
+
+AIGC 与身份模型在 UI 进程启动时分别进入独立常驻 worker，后续查询复用同一模型对象。依赖环境不能安全统一时仍保持进程隔离；常驻服务失败会降级为相同真实 runner 的一次性子进程，不会换成 Demo 或随机值。
+
+## 模型与算法摘要
+
+- AIGC：MobileNetV3-Small 空间二分类 + 87 维频域统计 LogisticRegression；`0.62 × 空间分数 + 0.38 × 频域分数`，图片阈值 `0.36360106194654707`。
+- 身份：InsightFace `buffalo_l`，512 维归一化 embedding、余弦相似度、按人物均值聚合、Top-5。
+- 底图：Haar 人脸屏蔽、HSV/灰度/边缘/外围全局特征粗排 Top-20，再用 SIFT/BFMatcher/Lowe 0.75/RANSAC；最终 `0.45 × global + 0.55 × local`。
+
+AIGC `score` 是 FAKE 风险分数；身份和底图 `score` 是各自检索空间中的相似度。三者都不是系统准确率，也不能相加。
+
+## 命令行
+
+只运行指定模块：
 
 ```powershell
-D:\python_3.14\python.exe -B check_environment.py
+D:\python_3.14\python.exe -B launcher.py --query "D:\path\query.jpg" --modules deepfake
+D:\python_3.14\python.exe -B launcher.py --query "D:\path\query.jpg" --modules identity source
 ```
 
-检查项只给出 `PASS`、`WARN` 或 `FAIL`。出现 `FAIL` 时先修正 `config.json` 或本机依赖。
-
-## 启动前端
-
-双击 `run_app.bat`，然后访问 `http://127.0.0.1:8766`。默认进入 `Live Analysis`：上传图片、确认身份库与底图库、点击“开始真实分析”。运行期间页面会持续显示创建任务、身份搜索、底图搜索和完成状态。
-
-## 命令行运行一张图片
-
-```bat
-run_case.bat "D:\path\query.jpg"
-```
-
-可选覆盖两个图库：
-
-```bat
-run_case.bat "D:\path\query.jpg" "D:\identity_library" "D:\source_library"
-```
-
-等价入口：
+默认三个模块全选：
 
 ```powershell
 D:\python_3.14\python.exe -B launcher.py --query "D:\path\query.jpg"
 ```
 
-终端会输出 task_id、task_dir、A/B 状态、A/B 耗时和总耗时。
+可用 `--identity-library` 和 `--source-library` 覆盖本次图库。空库或缺库在进入耗时搜索前直接失败。
 
-## 输出与 task
+## 任务归档
 
-结果位于 `runtime\tasks\case_YYYYMMDD_HHMMSS_xxxx\`。每次运行生成全新的目录，不会覆盖其他任务：
+结果位于 `runtime\tasks\case_YYYYMMDD_HHMMSS_xxxx\`：
 
 ```text
 query.jpg
+detection_result.json
 identity_result.json
 source_result.json
 unified_result.json
 task_status.json
 visualizations/
-logs/identity_stdout.log
-logs/identity_stderr.log
-logs/source_stdout.log
-logs/source_stderr.log
+logs/progress.json
+logs/*_stdout.log
+logs/*_stderr.log
+logs/*_execution.log
 ```
 
-若一个模块失败，另一模块仍继续；一个成功时 task 为 `partial_success`，两个都失败时为 `error`。
+`unified_result.json` 是证据汇总，不包含融合概率。原生 AIGC 报告和底图匹配可视化继续保存在该 task 内。
 
-## 验收结果边界
+## 检查与测试
 
-在 D 独立准备的第一阶段验收数据上，人脸身份搜索10/10案例正确身份均排名第1；原始底图搜索10/10案例真实底图均排名第1。
+只读环境/数据库检查：
 
-该结果仅代表当前限定闭集、小规模候选库测试条件，不代表真实公安场景总体准确率。
+```powershell
+D:\python_3.14\python.exe -B check_environment.py
+D:\python_3.14\python.exe -B system_status.py
+```
 
-本产品默认配置引用原始 45 张身份参考图与 203 张底图库；第1.5轮正式 profile 使用的是从同一来源物化的 44/200 选择集。三个集成回归案例在默认产品图库下的 Rank-1 与冻结正式结果一致，但不得据此扩大准确率结论。
+轻量工程回归：
 
-## 已知问题与后续接入
+```powershell
+D:\刑侦查竞赛\first_week\_round1_work\backend\envs\A_env\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
+cd frontend
+D:\python_3.14\python.exe -B -m unittest discover -s tests -v
+```
 
-当前限制见 `KNOWN_ISSUES.md`。后续如新增模块，应在 orchestrator 的 task 契约中增加独立子进程和统一状态，再由前端读取归档结果；不得把算法复制进 orchestrator，也不得修改冻结 A/B 的模型、特征、权重、阈值或排序。
+真实长回归（会加载模型并扫描完整底图库）：
 
+```powershell
+D:\python_3.14\python.exe -B tests\run_real_integration.py `
+  --query examples\integrated_query_001.jpg `
+  --query examples\integrated_query_003.jpg
+```
+
+测试范围、原始结果和边界见 `docs\REGRESSION_TEST_PLAN.md` 与最终 `docs\REGRESSION_TEST_REPORT.md`。
+
+## 外部资源与迁移
+
+为保证开发目录和 `releases` 正式目录都能运行，当前 `config.json` 明确引用本机已验证资源：
+
+- `D:\python\python.exe`；
+- `D:\刑侦查竞赛\first_week\...\A_env` 与 `B_env`；
+- `D:\刑侦查竞赛\first_week\data\identity_library`；
+- `D:\刑侦查竞赛\first_week\data\source_library`；
+- `C:\Users\lanji\AppData\Local\Programs\DeepfakeDetectionSystem`；
+- `C:\Users\lanji\.insightface\models\buffalo_l`。
+
+发布目录携带程序、冻结适配器和 MobileNet 官方基础权重，但不重复打包大图库、InsightFace 模型或安装版模型。换机时必须先安装/复制真实资源并更新 `config.json`，然后运行环境检查；不得用 Demo 文件替代后宣称真实可用。
+
+## 重要文档
+
+- `docs\AIGC_SOURCE_LOCATION_AUDIT.md`：源码定位与安装版对应证据。
+- `docs\MODEL_AND_RUNTIME.md`：模型、阈值、预处理和设备说明。
+- `docs\DATABASE_AUDIT.md`：图库、缓存和 index 审计。
+- `docs\DEPENDENCY_AUDIT.md`：环境冲突与隔离方案。
+- `docs\THREE_MODULE_ARCHITECTURE.md`：工程结构。
+- `docs\THREE_MODULE_GUIDE.md`：操作与结果解释。
+- `KNOWN_ISSUES.md`：当前限制。
+
+本系统用于风险筛查和侦查线索辅助，不替代原始介质固定、哈希校验、人工复核或司法鉴定。
```

### `check_environment.py`

```diff
--- base/check_environment.py
+++ target/check_environment.py
@@ -9,6 +9,8 @@
 from dataclasses import dataclass
 from pathlib import Path
 from typing import Any
+
+from system_status import inspect_system_status
 
 
 PRODUCT_ROOT = Path(__file__).resolve().parent
@@ -239,6 +241,8 @@
         return [CheckResult("FAIL", "config", f"无法读取 {resolved_config}：{exc}")]
 
     checks: list[CheckResult] = [
+        _check_file(config, resolved_config, "deepfake", "python", "AIGC Python"),
+        _check_file(config, resolved_config, "deepfake", "runner", "AIGC adapter"),
         _check_file(config, resolved_config, "identity", "python", "A Python"),
         _check_file(config, resolved_config, "source", "python", "B Python"),
         _check_file(config, resolved_config, "identity", "runner", "A runner"),
@@ -247,13 +251,35 @@
         _check_library(config, resolved_config, "source", "source_library"),
     ]
     checks.extend(_check_models(config, resolved_config))
+    system = inspect_system_status(resolved_config)
+    for module_name, item in (
+        ("deepfake", "AIGC EXE/模型/config/SHA"),
+        ("identity", "身份库/embedding/index"),
+        ("source", "底图库/实时扫描状态"),
+    ):
+        module = system[module_name]
+        checks.append(
+            CheckResult(
+                "PASS" if module.get("available") else "FAIL",
+                item,
+                str(module.get("message") or module.get("status")),
+            )
+        )
+    checks.append(
+        CheckResult(
+            "PASS",
+            "运算设备",
+            str(system.get("device", {}).get("message", "设备信息不可用"))
+            + f"；AIGC 实际设备={system.get('device', {}).get('deepfake_effective', 'unknown')}",
+        )
+    )
     checks.append(_check_frontend(config, resolved_config))
     checks.append(_check_runtime(config, resolved_config))
     return checks
 
 
 def build_parser() -> argparse.ArgumentParser:
-    parser = argparse.ArgumentParser(description="检查双搜索产品的必要运行环境。")
+    parser = argparse.ArgumentParser(description="检查三模块产品的必要运行环境。")
     parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="配置 JSON 路径")
     return parser
 
```

### `config.json`

```diff
--- base/config.json
+++ target/config.json
@@ -1,15 +1,43 @@
 {
-  "schema_version": "1.0",
-  "product_name": "face_source_trace_two_search_v0.1_official",
+  "schema_version": "2.0",
+  "product_name": "face_source_trace_three_module_v0.2",
   "runtime": {
     "tasks_root": "runtime/tasks",
     "inbox_root": "runtime/inbox",
-    "identity_cache": "runtime/cache/identity"
+    "identity_cache": "runtime/cache/identity",
+    "persistent_workers_enabled": true,
+    "persistent_startup_timeout_seconds": 180
+  },
+  "deepfake": {
+    "enabled": true,
+    "persistent_enabled": true,
+    "python": "D:\\python\\python.exe",
+    "python_args": ["-S", "-B"],
+    "runner": "{APP_ROOT}\\modules\\deepfake_detection\\run_deepfake_detection.py",
+    "install_root": "C:\\Users\\lanji\\AppData\\Local\\Programs\\DeepfakeDetectionSystem",
+    "install_root_read_only": true,
+    "torch_home": "{APP_ROOT}\\runtime\\cache\\deepfake\\torch",
+    "base_weight": "{APP_ROOT}\\runtime\\cache\\deepfake\\torch\\hub\\checkpoints\\mobilenet_v3_small-047dcff4.pth",
+    "device": "auto",
+    "timeout_seconds": 180,
+    "hide_console": true,
+    "extra_args": [],
+    "known_sha256": {
+      "executable": "bc67e52b77fa0559d4d6fe04f024dcdb4a8edf43128f80f7d70947e3da490203",
+      "spatial_model": "5bb4b33936eb3adc631828ea271a466d62070a2b8de1902e5d96122317534794",
+      "frequency_model": "2ec6828eb2a3151784667eb0c105f07cdb41de6ec2df587d28b2766ceee063eb",
+      "fusion_config": "8f76abeabb55da1119b28689e15ccd31eac7f1066e401e766987efbb4603e0b2",
+      "base_weight": "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f",
+      "adapter": "fd74aaa13996f735f9bbbe86afd6d2919c3bee5c0722c019c9f3792be6aeab25",
+      "adapter_runtime": "c6f3836d4c9252cbb2fc315acfd97e78b5f0b8bf10941e607bea996bcb86d836",
+      "adapter_inspector": "f2b46b0102afcca041f8a173ac9e9279548483632ba4932f25a4086eede03dba"
+    }
   },
   "identity": {
-    "python": "../first_week/_round1_work/backend/envs/A_env/Scripts/python.exe",
-    "runner": "modules/identity_search/run_identity_search.py",
-    "library": "../first_week/data/identity_library",
+    "persistent_enabled": true,
+    "python": "D:\\刑侦查竞赛\\first_week\\_round1_work\\backend\\envs\\A_env\\Scripts\\python.exe",
+    "runner": "{APP_ROOT}\\modules\\identity_search\\run_identity_search.py",
+    "library": "D:\\刑侦查竞赛\\first_week\\data\\identity_library",
     "top_k": 5,
     "timeout_seconds": 180,
     "ctx": -1,
@@ -17,15 +45,15 @@
     "offline": true
   },
   "source": {
-    "python": "../first_week/_round1_work/backend/envs/B_env/Scripts/python.exe",
-    "runner": "modules/source_search/run_source_search.py",
-    "library": "../first_week/data/source_library",
+    "python": "D:\\刑侦查竞赛\\first_week\\_round1_work\\backend\\envs\\B_env\\Scripts\\python.exe",
+    "runner": "{APP_ROOT}\\modules\\source_search\\run_source_search.py",
+    "library": "D:\\刑侦查竞赛\\first_week\\data\\source_library",
     "top_k": 5,
-    "timeout_seconds": 180
+    "timeout_seconds": 600
   },
   "frontend": {
     "python": "{CURRENT_PYTHON}",
     "address": "127.0.0.1",
-    "port": 8766
+    "port": 8767
   }
 }
```

### `docs/FRONTEND_VERIFICATION.md`

```diff
--- base/docs/FRONTEND_VERIFICATION.md
+++ target/docs/FRONTEND_VERIFICATION.md
@@ -1,32 +1,28 @@
-# 前端实机验证
+# 前端真实验收记录
 
-- 启动入口：`run_app.bat`
-- 解释器：`D:\python_3.14\python.exe`
-- 服务：`127.0.0.1:8766`
-- 健康端点：HTTP 200，`ok`
-- 主页：HTTP 200，包含 Streamlit 页面
+日期：2026-08-25  
+服务：`http://127.0.0.1:8767`  
+工具：Codex 应用内浏览器，真实 Streamlit 服务。
 
-真实 Live Analysis 页面驱动结果：
+## 验收结果
 
-```text
-task_id = case_20260820_162420_59ac
-status = success
-A = success, 5 candidates, Rank-1 person_001
-B = success, 5 candidates, Rank-1 122.jpg
-A runtime = 6.517758 s
-B runtime = 52.208761 s
-total runtime = 58.732793 s
-```
+- 页面标题为“AI换脸涉案图像智能鉴伪与原始素材溯源系统”。
+- 首页状态不是 mock：AIGC 可用；身份 9 人/45 图/43 embedding/index ready；底图 203 图/index not_persistent；实际设备 CPU。
+- 侧栏报告 AIGC/身份常驻服务 ready，模型可跨查询复用。
+- 未上传时四按钮禁用；上传受支持图片后全部启用。
+- 一键综合分析显示模型加载、图像检测、人脸/身份特征、身份库搜索、底图特征和底图库搜索进度。
+- `case_20260825_150652_5edf` 三模块均 success，总耗时 53.256 秒。
+- 证据汇总并列呈现 AIGC、身份 Top1、底图 Top1，并明确禁止跨模块融合概率。
+- AIGC、身份 Top-5、底图候选/可视化三个页签均能读取真实 task，并通过结果 schema 与本地资源路径安全检查。
+- 测试结束后临时 Streamlit 服务、AIGC worker 和 identity worker 均已退出；未触碰占用 8766 的现有双搜索程序。
 
-页面断言：
+## 截图
 
-- 默认模式为 Live Analysis。
-- 页面 exception=0、error=0。
-- 身份参考图 5/5 可加载。
-- 候选底图 5/5 可加载。
-- 匹配可视化 5/5 可加载。
-- 指标卡 13 个，结果标签页 3 个。
-- 上传临时文件在任务创建后自动删除。
+- `screenshots/three_module_home_ready.png`
+- `screenshots/three_module_live_progress.png`
+- `screenshots/three_module_live_result_summary.png`
+- `screenshots/three_module_identity_results.png`
+- `screenshots/three_module_source_results.png`
 
-服务验证完成后已正常停止。
+同一批截图的轻量副本位于 `test_results/ui/`，用于复核包。
 
```

### `frontend/README.md`

```diff
--- base/frontend/README.md
+++ target/frontend/README.md
@@ -1,64 +1,47 @@
-# C Frontend：轻量结果查看器
+# 三模块 Streamlit 前端
 
-这是第 1 轮规范化后的 Streamlit 前端组件。它保留原 C 的上传预览、库选择、动态候选、两类 Top-K、分数、路径、图片、匹配图和异常提示思路，同时移除了不可搬迁环境和运行残留。组件不 import、不调用 A/B，也不实现 launcher。
+前端以双搜索 official Streamlit UI 为基线，正式接入 AI 换脸鉴伪，产品标题为“AI换脸涉案图像智能鉴伪与原始素材溯源系统”。
 
-## 环境与安装
+## 交互与真实运行
 
-- 已验证环境：Python 3.14.3、Streamlit 1.58.0。
-- 声明范围：Python 3.10+，`streamlit>=1.54,<2.0`。
+`Live Analysis` 只有一个涉案图片上传入口，对应四个操作：
 
-```powershell
-python -m venv <outside-this-source-tree>\venv
-<outside-this-source-tree>\venv\Scripts\python.exe -m pip install -r requirements.txt
-```
+- `AI换脸鉴伪`：仅选择 `deepfake`；
+- `人脸身份搜索`：仅选择 `identity`；
+- `原始底图搜索`：仅选择 `source`；
+- `一键综合分析`：选择三个模块。
 
-虚拟环境必须建在源码目录之外，不要把 `.venv` 复制进交付包。
+前端调用 `orchestrator.run_task(..., selected_modules=..., services=...)`，未选模块由后端以 `status=skipped` 记录。一个模块失败不会抹去其他模块结果。
 
-## 启动
+顶部和侧栏使用 `system_status.inspect_system_status()` 的真实检查结果，显示 AIGC 模型、身份数据库、底图数据库和运算设备。显式空库、未初始化或不可用会立即禁用相应按钮，不进入无期限 loading。
+
+`RuntimeServiceManager` 通过 `@st.cache_resource` 在 UI 进程生命周期内启动一次，供 AIGC/身份模型复用。任一服务未就绪时，页面显示降级原因，调度器自动改用隔离子进程，UI 不因服务启动失败而崩溃。
+
+## 结果语义
+
+结果区固定保留三个页签：AI 换脸鉴伪、人脸身份搜索、原始底图搜索。上方的“证据汇总”只并列展示三类证据，绝不计算跨任务融合概率。
+
+AIGC 原生 `LOW_CONFIDENCE` 和 `INSUFFICIENT_EVIDENCE` 保持“不确定”；`is_deepfake=null` 绝不会被映射成 REAL。`score` 只按原模块的 FAKE 风险分数语义显示，不宣称是系统准确率或司法结论。
+
+## Task Result、Mock Demo 与路径安全
+
+- `Task Result` 读取任务根下固定命名的 `detection_result.json`、`identity_result.json` 和 `source_result.json`。
+- 旧双模块 task 如缺少 `detection_result.json`，会合成明确的 AIGC `skipped` 占位，不伪造结果。
+- `Mock Demo` 只验证三页签、汇总布局和路径解析；其 SVG/JSON 不是模型或准确率证据。
+- 资源路径仅允许位于 task、显式批准的库根或 sample 根；拒绝 `..`、越界绝对路径、驱动器相对路径和越界符号链接。
+- official 的“关闭系统”保留，并会先幂等关闭持久服务。
+
+完整 JSON 契约见 [TASK_DIRECTORY_CONTRACT.md](TASK_DIRECTORY_CONTRACT.md)。
+
+## 启动与轻量验证
 
 ```powershell
 python -m streamlit run app.py --server.fileWatcherType none --browser.gatherUsageStats false
 ```
 
-## Mock Demo
-
-选择 `Mock Demo` 后点击“加载 Mock 演示”。内置两份 JSON 都有 5 条动态候选，所有 query、身份参考图、底图和匹配图引用均存在。SVG 占位图仅用于前端布局测试，不是算法素材或准确率证据，也不使用 D 数据。
-
-上传控件只做浏览器会话内预览，不写入 `runtime_inputs` 或任何源码目录。关闭/刷新会话后，前端不保留上传文件。
-
-## Task Result
-
-选择 `Task Result`，明确填写：
-
-1. `task_dir`；
-2. 允许读取的身份库根；
-3. 允许读取的底图库根；
-4. 可选的额外批准资源根。
-
-点击“读取 Task Result”后，组件只读取 task 根下固定命名的两份 JSON。仓库内 `mock_task` 是无需后端的 Task 模式演示。完整契约见 [TASK_DIRECTORY_CONTRACT.md](TASK_DIRECTORY_CONTRACT.md)。
-
-## JSON 与安全规则
-
-加载器严格验证：顶层字段、状态、候选数组、候选必需字段、rank 正整数与连续顺序、数值字段的有限数值类型、路径字段字符串类型，以及 error 状态语义。非法 JSON、错误 schema、空候选和 `status=error` 都会形成受控页面状态，不向外抛出页面异常。
-
-资源按字段限制在当前 task、明确批准的身份/底图库或 sample 根内。`..`、越界绝对路径、驱动器相对路径和越界符号链接会在 `st.image` 之前被拒绝。用户显式填写一个库根表示授权本次页面读取该目录；不要批准不可信的宽泛目录。
-
-## 自动测试
-
-不依赖 pytest，使用标准库 unittest 与 Streamlit AppTest：
+不启动真实模型的前端单元测试：
 
 ```powershell
 $env:PYTHONDONTWRITEBYTECODE='1'
 python -B -m unittest discover -s tests -v
 ```
-
-覆盖 Mock 完整性、身份/底图 view-model、非法 JSON、严格 schema、`status=error`、空候选、路径越界、Task 模式与 Mock/Task 页面级 smoke。HTTP 200 的独立进程验证结果记录在 `tests_output`。
-
-## 已知限制
-
-- 本组件只展示已有结果，不负责启动、轮询、超时、取消或清理算法任务。
-- 上传仅预览；未来 launcher 负责把正式 query 写入统一 runtime/task 目录。
-- task 根和库根由操作者显式提供；生产 launcher 应通过配置传入更窄且只读的批准根。
-- SVG Mock 只验证布局与路径，不代表真实 JPEG/MPO 解码链或算法效果。
-- 本轮未运行 A/B，更未使用 D 数据做正式独立验收或调参。
-
```

### `frontend/TASK_DIRECTORY_CONTRACT.md`

```diff
--- base/frontend/TASK_DIRECTORY_CONTRACT.md
+++ target/frontend/TASK_DIRECTORY_CONTRACT.md
@@ -1,27 +1,44 @@
-# C 前端任务目录契约
+# 前端任务目录契约 v2
 
-本组件只读取任务结果，不创建任务、不调用 A/B。未来 launcher 每次运行应建立独立目录：
+每次上传建立独立任务目录：
 
 ```text
-task_xxx/
+case_xxx/
 ├── query.jpg
+├── detection_result.json
 ├── identity_result.json
 ├── source_result.json
+├── unified_result.json
+├── task_status.json
 ├── visualizations/
-│   ├── identity/
-│   └── source/
-├── stdout/
-├── stderr/
-└── task_status.json
+└── logs/
 ```
 
-## 固定结果文件
+三份结果 JSON 是 task 根的直接子文件、UTF-8 编码，单文件不超过 5 MiB。状态允许 `success`、`error` 和 `skipped`；`skipped` 明确表示本次未选择模块，不是成功或失败结论。
 
-- `identity_result.json` 与 `source_result.json` 必须是 task 根目录的直接子文件，UTF-8 编码，单文件不超过 5 MiB。
-- 顶层字段固定为 `status`、`query_image`、`candidates`、`runtime_seconds`、`error_message`。
-- `status` 只允许 `success` 或 `error`。`error` 必须有非空错误消息和空候选数组。
-- `runtime_seconds` 是非负有限数值；它是模块口径，不等同于 launcher 的端到端 wall time。
-- 候选 `rank` 必须按数组顺序连续为 `1..N`。
+## AI 换脸鉴伪结果
+
+必需字段：
+
+```text
+status, query_image, is_deepfake, score, label,
+runtime_seconds, details, error_message
+```
+
+- `is_deepfake` 是布尔值或 null。
+- `score` 是 0..1 的有限数值或 null，只按原模块的 FAKE 风险分数语义显示。
+- `label` 是字符串或 null；`details` 必须是对象。
+- `status=success` 且 `label=low_confidence/insufficient_evidence`、`is_deepfake=null` 是不确定结果，不是 REAL。
+- `status=error` 必须有非空 `error_message`。
+- `status=skipped` 必须使 `is_deepfake=null`、`score=null`，`label` 只能为 null 或 `skipped`。
+
+## 人脸身份与原始底图结果
+
+两份搜索结果的顶层字段：
+
+```text
+status, query_image, candidates, runtime_seconds, error_message
+```
 
 身份候选字段：
 
@@ -35,14 +52,18 @@
 rank, file_path, global_score, local_score, final_score, visualization_path
 ```
 
-## 路径规则
+- 候选 `rank` 必须按数组顺序连续为 `1..N`。
+- `status=error` 必须有非空错误消息及空候选数组。
+- `status=skipped` 必须有空候选数组，可额外携带 `details.reason`。
+- `runtime_seconds` 是非负有限数值，为模块口径。
 
-- 推荐 `query_image` 和 `visualization_path` 相对于 `task_dir`。
-- `reference_image` 可相对于显式批准的身份库根。
-- `file_path` 可相对于显式批准的底图库根。
-- JSON 也可给出绝对路径，但它必须仍位于 task 或用户显式批准的资源根内。
-- 同时兼容 `/` 与 `\`，但拒绝 `..`、驱动器相对路径、相对路径冒号、越界绝对路径和越界符号链接。
-- 路径通过安全检查不代表文件一定存在；文件在加载后被删除时，页面显示警告而不崩溃。
+## 旧双模块任务
 
-每个 query 必须使用独立 task 目录，避免 JSON、日志和 `match_*.jpg` 互相覆盖。
+为保持 official 历史 task 可读，如任务目录有合法的 `identity_result.json` 和 `source_result.json`，但没有 `detection_result.json`，加载器会在内存中生成一份明确的 AIGC `skipped` 占位。原任务文件不会被改写。
 
+## 路径安全
+
+- `query_image` 和匹配可视化应位于 task 根。
+- 身份参考图、底图可位于操作者明确批准的对应库根。
+- 拒绝 `..`、驱动器相对路径、相对路径冒号、越界绝对路径和越界符号链接。
+- 路径通过边界校验后若文件被删除，页面显示受控警告而不崩溃。
```

### `frontend/app.py`

```diff
--- base/frontend/app.py
+++ target/frontend/app.py
@@ -1,5 +1,6 @@
 from __future__ import annotations
 
+from collections.abc import Collection, Mapping
 import os
 from pathlib import Path
 import secrets
@@ -27,12 +28,23 @@
 from components.data_loader import (
     ResultDataError,
     TaskResults,
+    load_detection_result,
     load_identity_result,
     load_source_result,
     load_task_results,
 )
 from components.path_security import ResourcePathPolicy
-from components.result_views import render_identity_results, render_source_results
+from components.result_views import (
+    render_deepfake_result,
+    render_evidence_summary,
+    render_identity_results,
+    render_source_results,
+)
+from components.status_adapter import (
+    ModuleAvailability,
+    device_label,
+    module_availability,
+)
 
 
 PROJECT_ROOT = FRONTEND_ROOT
@@ -41,6 +53,14 @@
 MOCK_TASK_ROOT = FRONTEND_ROOT / "mock_task"
 IDENTITY_LIBRARY_ROOT = SAMPLE_ROOT / "identity_libraries"
 SOURCE_LIBRARY_ROOT = SAMPLE_ROOT / "source_libraries"
+MODULE_KEYS = ("deepfake", "identity", "source")
+MODULE_LABELS = {
+    "deepfake": "AI 换脸鉴伪",
+    "identity": "人脸身份搜索",
+    "source": "原始底图搜索",
+}
+MAX_UPLOAD_BYTES = 20 * 1024 * 1024
+ALLOWED_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
 
 
 def discover_libraries(library_root: Path) -> list[Path]:
@@ -62,20 +82,105 @@
     return (path,)
 
 
+@st.cache_data(ttl=5, show_spinner=False)
+def _cached_system_status(config_path_text: str) -> dict[str, Any]:
+    """Inspect real resources while turning import/inspection failures into data."""
+
+    try:
+        from system_status import inspect_system_status
+
+        snapshot = inspect_system_status(config_path=config_path_text)
+        if not isinstance(snapshot, dict):
+            raise TypeError("系统状态检查结果不是 JSON 对象")
+        return {"snapshot": snapshot, "error": None}
+    except Exception as exc:
+        return {
+            "snapshot": {
+                "overall_status": "unavailable",
+                "device": "unknown",
+                "deepfake": {
+                    "available": False,
+                    "message": "无法完成 AIGC 资源检查。",
+                },
+                "identity": {
+                    "available": False,
+                    "message": "无法完成身份库检查。",
+                },
+                "source": {
+                    "available": False,
+                    "message": "无法完成底图库检查。",
+                },
+            },
+            "error": f"{type(exc).__name__}: {exc}",
+        }
+
+
+@st.cache_resource(show_spinner="正在初始化 AIGC 与身份模型服务……")
+def _cached_runtime_services(config_path_text: str) -> dict[str, Any]:
+    """Start model-caching workers once; callers can always fall back safely."""
+
+    manager: Any = None
+    try:
+        from runtime_services import RuntimeServiceManager
+
+        manager = RuntimeServiceManager(config_path_text)
+        startup_status = manager.start()
+        if not isinstance(startup_status, dict):
+            startup_status = manager.get_status()
+        return {
+            "manager": manager,
+            "status": startup_status,
+            "error": None,
+        }
+    except Exception as exc:
+        if manager is not None:
+            try:
+                manager.close()
+            except Exception:
+                pass
+        return {
+            "manager": None,
+            "status": {"started": False, "ready": False, "health": "error"},
+            "error": f"{type(exc).__name__}: {exc}",
+        }
+
+
+def _runtime_service_snapshot(bundle: Mapping[str, Any]) -> dict[str, Any]:
+    manager = bundle.get("manager")
+    if manager is not None:
+        try:
+            status = manager.get_status()
+            if isinstance(status, dict):
+                return status
+        except Exception as exc:
+            return {
+                "started": False,
+                "ready": False,
+                "health": "error",
+                "error": f"{type(exc).__name__}: {exc}",
+            }
+    status = bundle.get("status")
+    return dict(status) if isinstance(status, Mapping) else {}
+
+
 def _store_results(
     *,
     mode: str,
+    deepfake_result: dict[str, Any],
     identity_result: dict[str, Any],
     source_result: dict[str, Any],
     path_policy: ResourcePathPolicy,
     source_label: str,
+    legacy_dual_module: bool = False,
 ) -> None:
     st.session_state["loaded_results"] = {
         "mode": mode,
+        "deepfake": deepfake_result,
         "identity": identity_result,
         "source": source_result,
         "policy": path_policy,
         "source_label": source_label,
+        "legacy_dual_module": legacy_dual_module,
     }
 
 
@@ -85,12 +190,16 @@
         source_roots=(source_library,),
         sample_roots=(SAMPLE_ROOT,),
     )
+    deepfake = load_detection_result(
+        MOCK_RESULTS_ROOT / "detection_result.json", policy
+    )
     identity = load_identity_result(
         MOCK_RESULTS_ROOT / "identity_result.json", policy
     )
     source = load_source_result(MOCK_RESULTS_ROOT / "source_result.json", policy)
     _store_results(
         mode="Mock Demo",
+        deepfake_result=deepfake,
         identity_result=identity,
         source_result=source,
         path_policy=policy,
@@ -115,10 +224,12 @@
     )
     _store_results(
         mode=mode,
+        deepfake_result=results.deepfake_result,
         identity_result=results.identity_result,
         source_result=results.source_result,
         path_policy=results.path_policy,
         source_label=str(results.task_dir),
+        legacy_dual_module=results.legacy_dual_module,
     )
 
 
@@ -130,42 +241,123 @@
     return str(identity_root), str(source_root), inbox_root
 
 
+def _validate_upload(uploaded_query: Any) -> str:
+    if uploaded_query is None:
+        raise ResultDataError("请先上传查询图片。")
+    size = getattr(uploaded_query, "size", 0)
+    if isinstance(size, (int, float)) and size > MAX_UPLOAD_BYTES:
+        raise ResultDataError("查询图片超过 20 MiB 上限。")
+    suffix = Path(str(uploaded_query.name)).suffix.lower()
+    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
+        raise ResultDataError("查询图片扩展名只允许 jpg/jpeg/png/webp。")
+    return suffix
+
+
+def _humanize_progress(event: Any) -> str:
+    if isinstance(event, Mapping):
+        message = event.get("message")
+        if isinstance(message, str) and message.strip():
+            return message.strip()
+        module = str(event.get("module") or event.get("service") or "").lower()
+        stage = str(event.get("stage") or event.get("state") or "").lower()
+        if module == "deepfake":
+            return "正在加载 AIGC 模型并检测图像……"
+        if module == "identity":
+            return "正在检测人脸、提取身份特征并搜索数据库……"
+        if module == "source":
+            return "正在提取底图特征并搜索数据库……"
+        if stage:
+            return f"任务进度：{stage}"
+        return "正在执行分析任务……"
+    message = str(event).strip()
+    replacements = {
+        "正在进行人脸身份搜索": "正在检测人脸、提取身份特征并搜索身份数据库",
+        "正在进行原始底图搜索": "正在提取底图特征并搜索底图数据库",
+        "正在进行换脸鉴伪": "正在加载 AIGC 模型并检测图像",
+    }
+    for original, friendly in replacements.items():
+        if original in message:
+            return message.replace(original, friendly)
+    return message or "正在执行分析任务……"
+
+
 def _run_live_analysis(
     uploaded_query: Any,
     identity_root_text: str,
     source_root_text: str,
+    selected_modules: Collection[str],
+    services: Any | None,
 ) -> None:
-    if uploaded_query is None:
-        raise ResultDataError("请先上传查询图片。")
-    if getattr(uploaded_query, "size", 0) > 20 * 1024 * 1024:
-        raise ResultDataError("查询图片超过 20 MiB 上限。")
-    suffix = Path(uploaded_query.name).suffix.lower()
-    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
-        raise ResultDataError("查询图片扩展名只允许 jpg/jpeg/png/webp。")
-
-    identity_root = _parse_optional_root(identity_root_text, "身份库根")
-    source_root = _parse_optional_root(source_root_text, "底图库根")
-    if not identity_root or not source_root:
-        raise ResultDataError("Live Analysis 必须填写身份库和底图库。")
+    modules = frozenset(selected_modules)
+    if not modules or not modules <= set(MODULE_KEYS):
+        raise ResultDataError(f"未知模块选择：{sorted(modules)}")
+    suffix = _validate_upload(uploaded_query)
+
+    identity_root = (
+        _parse_optional_root(identity_root_text, "身份库")
+        if "identity" in modules
+        else ()
+    )
+    source_root = (
+        _parse_optional_root(source_root_text, "底图库")
+        if "source" in modules
+        else ()
+    )
+    if "identity" in modules and not identity_root:
+        raise ResultDataError("人脸身份搜索必须指定身份库。")
+    if "source" in modules and not source_root:
+        raise ResultDataError("原始底图搜索必须指定底图库。")
+
     _, _, inbox_root = _live_defaults()
     inbox_root.mkdir(parents=True, exist_ok=True)
     inbox_path = inbox_root / f"upload_{secrets.token_hex(8)}{suffix}"
     inbox_path.write_bytes(uploaded_query.getvalue())
+    status: dict[str, Any] | None = None
 
     try:
-        with st.status("正在创建独立任务…", expanded=True) as live_status:
-            status = run_task(
-                inbox_path,
-                identity_library=identity_root[0],
-                source_library=source_root[0],
-                config_path=DEFAULT_CONFIG,
-                progress_callback=live_status.write,
-            )
-            if status["status"] == "success":
-                live_status.update(label="真实分析完成", state="complete", expanded=False)
+        with st.status("正在创建独立任务……", expanded=True) as live_status:
+            selected_text = "、".join(MODULE_LABELS[key] for key in MODULE_KEYS if key in modules)
+            live_status.write(f"已选模块：{selected_text}")
+            if "deepfake" in modules:
+                live_status.write("正在准备 AIGC 模型与图像预处理……")
+            if "identity" in modules:
+                live_status.write("将执行人脸检测、身份特征提取与库内搜索。")
+            if "source" in modules:
+                live_status.write("将执行底图特征提取与库内搜索。")
+            try:
+                status = run_task(
+                    inbox_path,
+                    identity_library=identity_root[0] if identity_root else None,
+                    source_library=source_root[0] if source_root else None,
+                    selected_modules=modules,
+                    services=services,
+                    config_path=DEFAULT_CONFIG,
+                    progress_callback=lambda event: live_status.write(
+                        _humanize_progress(event)
+                    ),
+                )
+            except Exception:
+                live_status.update(
+                    label="分析任务未完成",
+                    state="error",
+                    expanded=True,
+                )
+                raise
+
+            overall = str(status.get("status") or "error")
+            if overall == "success":
+                live_status.write("分析完成。")
+                live_status.update(label="分析完成", state="complete", expanded=False)
+            elif overall == "partial_success":
+                live_status.write("所选模块部分成功；请查看下方错误与结果。")
+                live_status.update(
+                    label="分析部分完成",
+                    state="complete",
+                    expanded=True,
+                )
             else:
                 live_status.update(
-                    label=f"分析完成：{status['status']}",
+                    label=f"分析完成：{overall}",
                     state="error",
                     expanded=True,
                 )
@@ -175,14 +367,19 @@
         except OSError:
             pass
 
+    if not isinstance(status, dict) or not status.get("task_dir"):
+        raise ResultDataError("调度器未返回可读的 task_dir。")
     _load_task(
-        status["task_dir"],
-        str(identity_root[0]),
-        str(source_root[0]),
+        str(status["task_dir"]),
+        str(identity_root[0]) if identity_root else "",
+        str(source_root[0]) if source_root else "",
         "",
         mode="Live Analysis",
     )
     st.session_state["live_task_status"] = status
+    st.session_state["live_selected_modules"] = tuple(
+        key for key in MODULE_KEYS if key in modules
+    )
 
 
 def _render_loaded_results(mode: str) -> None:
@@ -192,53 +389,335 @@
 
     st.success(f"{mode} 结果已通过 schema 与路径安全校验。")
     st.write(f"**结果来源：** `{loaded['source_label']}`")
-    with st.expander("本次批准的资源根"):
-        for root in loaded["policy"].display_roots():
-            st.code(root)
-
-    identity_tab, source_tab, json_tab = st.tabs(
-        ["身份 Top 候选", "底图 Top 候选", "原始 JSON"]
-    )
+    if loaded.get("legacy_dual_module"):
+        st.warning(
+            "检测到旧版双模块 task：缺少 detection_result.json，"
+            "AIGC 页签明确标记为“未运行”，不伪造鉴伪结果。"
+        )
+
+    render_evidence_summary(
+        loaded["deepfake"], loaded["identity"], loaded["source"]
+    )
+    deepfake_tab, identity_tab, source_tab = st.tabs(
+        ["AI 换脸鉴伪", "人脸身份搜索", "原始底图搜索"]
+    )
+    with deepfake_tab:
+        render_deepfake_result(loaded["deepfake"])
     with identity_tab:
         render_identity_results(loaded["identity"], loaded["policy"])
     with source_tab:
         render_source_results(loaded["source"], loaded["policy"])
-    with json_tab:
-        st.markdown("#### 身份搜索 JSON")
+
+    with st.expander("原始 JSON 与批准资源根"):
+        st.markdown("#### AI 换脸鉴伪 JSON")
+        st.json(loaded["deepfake"])
+        st.markdown("#### 人脸身份搜索 JSON")
         st.json(loaded["identity"])
-        st.markdown("#### 底图搜索 JSON")
+        st.markdown("#### 原始底图搜索 JSON")
         st.json(loaded["source"])
+        st.markdown("#### 本次批准的资源根")
+        for root in loaded["policy"].display_roots():
+            st.code(root)
+
+
+def _render_status_strip(
+    snapshot: Mapping[str, Any],
+) -> tuple[dict[str, ModuleAvailability], tuple[str, str]]:
+    modules = {key: module_availability(snapshot, key) for key in MODULE_KEYS}
+    device = device_label(snapshot)
+    columns = st.columns(4)
+    for column, key in zip(columns[:3], MODULE_KEYS, strict=True):
+        item = modules[key]
+        column.metric(item.title, item.value)
+        column.caption(item.caption)
+    columns[3].metric("运算设备", device[0])
+    columns[3].caption(device[1])
+    return modules, device
+
+
+def _render_sidebar_system_status(
+    modules: Mapping[str, ModuleAvailability],
+    device: tuple[str, str],
+    service_bundle: Mapping[str, Any],
+    inspection_error: str | None,
+) -> None:
+    st.header("系统就绪状态")
+    for key in MODULE_KEYS:
+        item = modules[key]
+        icon = "🟢" if item.runnable else "🟠" if item.empty else "🔴"
+        st.write(f"{icon} **{item.title}**：{item.value}")
+        st.caption(item.caption)
+    st.write(f"🖥️ **运算设备**：{device[0]}")
+
+    service_status = _runtime_service_snapshot(service_bundle)
+    service_ready = {
+        key: bool((service_status.get(key) or {}).get("ready"))
+        for key in ("deepfake", "identity")
+        if isinstance(service_status.get(key), Mapping)
+    }
+    if service_bundle.get("manager") is None:
+        st.warning("模型缓存服务未启动；真实分析将降级为隔离子进程。")
+    elif service_ready and all(service_ready.values()):
+        st.success("AIGC/身份模型已由持久服务缓存，后续查询可复用。")
+    else:
+        ready_names = [MODULE_LABELS[key] for key, ready in service_ready.items() if ready]
+        ready_text = "、".join(ready_names) if ready_names else "无"
+        st.warning(
+            f"持久服务部分可用（已就绪：{ready_text}）；"
+            "不可用模块将自动降级为隔离子进程。"
+        )
+    if service_bundle.get("error"):
+        st.caption(f"服务降级原因：{service_bundle['error']}")
+    if inspection_error:
+        st.error(f"系统状态检查失败：{inspection_error}")
+    with st.expander("服务运行详情"):
+        st.json(service_status)
+
+
+def _module_meta(task_status: Mapping[str, Any], key: str) -> dict[str, Any]:
+    value = task_status.get(key)
+    return dict(value) if isinstance(value, Mapping) else {"status": "skipped"}
+
+
+def _render_live_summary() -> None:
+    task_status = st.session_state.get("live_task_status")
+    loaded = st.session_state.get("loaded_results")
+    if not isinstance(task_status, Mapping) or not loaded:
+        return
+    if loaded.get("mode") != "Live Analysis":
+        return
+
+    columns = st.columns(4)
+    columns[0].metric("任务总状态", str(task_status.get("status") or "未知"))
+    for column, key in zip(columns[1:], MODULE_KEYS, strict=True):
+        meta = _module_meta(task_status, key)
+        status = str(meta.get("status") or meta.get("state") or "skipped")
+        runtime = meta.get("runtime_seconds")
+        delta = f"{float(runtime):.3f} 秒" if isinstance(runtime, (int, float)) else None
+        column.metric(MODULE_LABELS[key], status, delta, delta_color="off")
+        execution_mode = meta.get("execution_mode")
+        fallback_reason = meta.get("fallback_reason")
+        if fallback_reason:
+            column.caption(f"已降级为隔离子进程：{fallback_reason}")
+        elif execution_mode:
+            column.caption(f"执行方式：{execution_mode}")
+        error = meta.get("error") or meta.get("error_message")
+        if status == "error" and error:
+            column.caption(str(error))
+    if task_status.get("task_id"):
+        st.write(f"**task_id：** `{task_status['task_id']}`")
+    total_runtime = task_status.get("runtime_seconds")
+    if isinstance(total_runtime, (int, float)):
+        st.write(f"**总耗时：** `{float(total_runtime):.3f} 秒`")
+    errors = task_status.get("errors")
+    if isinstance(errors, list):
+        for error in errors:
+            st.error(str(error))
+
+
+def _render_shutdown(service_bundle: Mapping[str, Any]) -> None:
+    st.divider()
+    if st.button("关闭系统", help="停止本地服务并退出程序", width="stretch"):
+        manager = service_bundle.get("manager")
+        if manager is not None:
+            try:
+                manager.close()
+            except Exception as exc:
+                st.warning(f"持久服务关闭时返回警告：{exc}")
+        st.success("系统正在关闭，可以关闭此浏览器页面。")
+        threading.Timer(
+            0.8,
+            lambda: os.kill(os.getpid(), signal.SIGTERM),
+        ).start()
+        st.stop()
+
+
+def _render_live_mode(
+    *,
+    modules: Mapping[str, ModuleAvailability],
+    service_bundle: Mapping[str, Any],
+    identity_root_text: str,
+    source_root_text: str,
+    live_config_error: str,
+) -> None:
+    st.subheader("案件图像分析")
+    st.caption("上传一张图片，然后选择单一模块或一键综合分析。")
+    uploaded_query = st.file_uploader(
+        "上传涉案图片",
+        type=["jpg", "jpeg", "png", "webp"],
+        accept_multiple_files=False,
+        help="上传文件只暂存于 runtime/inbox，调度器为每次分析创建独立 task_dir。",
+        key="live_query",
+    )
+    if uploaded_query is not None:
+        st.image(
+            uploaded_query,
+            caption=f"待分析：{uploaded_query.name}",
+            width=420,
+        )
+    else:
+        st.info("请先上传 JPG/JPEG/PNG/WEBP 图片（最大 20 MiB）。")
+
+    upload_error = ""
+    if uploaded_query is not None:
+        try:
+            _validate_upload(uploaded_query)
+        except ResultDataError as exc:
+            upload_error = str(exc)
+            st.error(upload_error)
+    upload_missing = uploaded_query is None
+    upload_blocked = upload_missing or bool(upload_error)
+    config_blocked = bool(live_config_error)
+    deepfake_disabled = upload_blocked or config_blocked or not modules["deepfake"].runnable
+    identity_disabled = upload_blocked or config_blocked or not modules["identity"].runnable
+    source_disabled = upload_blocked or config_blocked or not modules["source"].runnable
+    comprehensive_disabled = (
+        upload_blocked
+        or config_blocked
+        or any(not modules[key].runnable for key in MODULE_KEYS)
+    )
+
+    columns = st.columns(4)
+    deepfake_clicked = columns[0].button(
+        "AI换脸鉴伪",
+        type="primary",
+        disabled=deepfake_disabled,
+        width="stretch",
+    )
+    identity_clicked = columns[1].button(
+        "人脸身份搜索",
+        type="primary",
+        disabled=identity_disabled,
+        width="stretch",
+    )
+    source_clicked = columns[2].button(
+        "原始底图搜索",
+        type="primary",
+        disabled=source_disabled,
+        width="stretch",
+    )
+    comprehensive_clicked = columns[3].button(
+        "一键综合分析",
+        type="primary",
+        disabled=comprehensive_disabled,
+        width="stretch",
+    )
+
+    for key in MODULE_KEYS:
+        item = modules[key]
+        if item.empty:
+            st.error(f"{item.title}为空或未初始化：{item.message}；相应按钮已禁用。")
+        elif not item.available:
+            st.warning(f"{item.title}不可用：{item.message}；相应按钮已禁用。")
+    if live_config_error:
+        st.error(f"配置读取失败：{live_config_error}")
+
+    selected: frozenset[str] | None = None
+    if deepfake_clicked:
+        selected = frozenset({"deepfake"})
+    elif identity_clicked:
+        selected = frozenset({"identity"})
+    elif source_clicked:
+        selected = frozenset({"source"})
+    elif comprehensive_clicked:
+        selected = frozenset(MODULE_KEYS)
+
+    if selected is not None:
+        st.session_state.pop("loaded_results", None)
+        st.session_state.pop("live_task_status", None)
+        try:
+            _run_live_analysis(
+                uploaded_query,
+                identity_root_text,
+                source_root_text,
+                selected,
+                service_bundle.get("manager"),
+            )
+        except (ResultDataError, OrchestrationError, OSError, RuntimeError) as exc:
+            st.error(f"分析失败：{exc}")
+        except Exception as exc:
+            st.error(f"分析失败：{type(exc).__name__}: {exc}")
+
+    _render_live_summary()
+    _render_loaded_results("Live Analysis")
 
 
 def main() -> None:
-    st.set_page_config(page_title="换脸寻源双搜索系统 v1.0", page_icon="🔍", layout="wide")
-    st.title("换脸寻源双搜索系统 v1.0")
+    st.set_page_config(
+        page_title="AI换脸涉案图像智能鉴伪与原始素材溯源系统",
+        page_icon="🔍",
+        layout="wide",
+    )
+    st.markdown(
+        """
+        <style>
+        [data-testid="stMetric"] {
+            border: 1px solid rgba(128, 128, 128, 0.22);
+            border-radius: 0.75rem;
+            padding: 0.7rem 0.85rem;
+        }
+        </style>
+        """,
+        unsafe_allow_html=True,
+    )
+    st.title("AI换脸涉案图像智能鉴伪与原始素材溯源系统")
     st.caption(
-        "AI 换脸涉案图像智能鉴伪与原始素材溯源——身份搜索与原始底图搜索。"
-    )
+        "单图任务、三模块故障隔离；综合分析只汇总证据，不伪造融合概率。"
+    )
+
+    status_result = _cached_system_status(str(DEFAULT_CONFIG))
+    snapshot = status_result["snapshot"]
+    modules, device = _render_status_strip(snapshot)
+    service_bundle = _cached_runtime_services(str(DEFAULT_CONFIG))
 
     identity_libraries = discover_libraries(IDENTITY_LIBRARY_ROOT)
     source_libraries = discover_libraries(SOURCE_LIBRARY_ROOT)
 
     with st.sidebar:
+        _render_sidebar_system_status(
+            modules,
+            device,
+            service_bundle,
+            status_result.get("error"),
+        )
+        if st.button("重新检测 / 重试服务", width="stretch"):
+            manager = service_bundle.get("manager")
+            if manager is not None:
+                try:
+                    manager.close()
+                except Exception:
+                    pass
+            _cached_runtime_services.clear()
+            _cached_system_status.clear()
+            st.rerun()
+
+        st.divider()
         st.header("结果来源")
         mode = st.radio("运行模式", ["Live Analysis", "Task Result", "Mock Demo"])
 
         if mode == "Mock Demo":
             st.info("占位图片仅用于前端布局测试，不是算法素材或准确率证据。")
-            uploaded_query = st.file_uploader(
-                "可选：内存预览查询图片",
-                type=["jpg", "jpeg", "png", "webp"],
-                accept_multiple_files=False,
-                help="文件不会写入源码目录。",
-            )
             identity_names = [path.name for path in identity_libraries]
             source_names = [path.name for path in source_libraries]
-            selected_identity = st.selectbox("选择身份库", identity_names)
-            selected_source = st.selectbox("选择底图库", source_names)
-            load_button = st.button("加载 Mock 演示", type="primary")
+            selected_identity = st.selectbox(
+                "选择演示身份库",
+                identity_names or ["(无演示库)"],
+                disabled=not identity_names,
+            )
+            selected_source = st.selectbox(
+                "选择演示底图库",
+                source_names or ["(无演示库)"],
+                disabled=not source_names,
+            )
+            load_button = st.button(
+                "加载 Mock 演示",
+                type="primary",
+                disabled=not identity_names or not source_names,
+                width="stretch",
+            )
+            task_dir_text = identity_root_text = source_root_text = resource_root_text = ""
+            live_config_error = ""
         elif mode == "Task Result":
-            uploaded_query = None
             task_dir_text = st.text_input("task_dir", value=str(MOCK_TASK_ROOT))
             identity_root_text = st.text_input(
                 "批准的身份库根",
@@ -252,75 +731,67 @@
                 "额外批准资源根（可留空）",
                 value="",
             )
-            load_button = st.button("读取 Task Result", type="primary")
+            load_button = st.button("读取 Task Result", type="primary", width="stretch")
+            selected_identity = selected_source = ""
+            live_config_error = ""
         else:
             try:
                 default_identity, default_source, _ = _live_defaults()
                 live_config_error = ""
-            except (OrchestrationError, KeyError) as exc:
+            except (OrchestrationError, KeyError, TypeError) as exc:
                 default_identity, default_source = "", ""
                 live_config_error = str(exc)
-            uploaded_query = st.file_uploader(
-                "上传查询图片",
-                type=["jpg", "jpeg", "png", "webp"],
-                accept_multiple_files=False,
-                help="临时上传只进入 runtime/inbox；任务使用独立复制的 query.jpg。",
-                key="live_query",
-            )
             identity_root_text = st.text_input(
-                "身份库",
+                "本次任务身份库",
                 value=default_identity,
                 key="live_identity_root",
             )
             source_root_text = st.text_input(
-                "底图库",
+                "本次任务底图库",
                 value=default_source,
                 key="live_source_root",
             )
-            if live_config_error:
-                st.error(f"配置读取失败：{live_config_error}")
-            load_button = st.button(
-                "开始真实分析",
-                type="primary",
-                disabled=bool(live_config_error),
-            )
-
-        st.divider()
-        if st.button("关闭系统", help="停止本地服务并退出程序"):
-            st.success("系统正在关闭，可以关闭此浏览器页面。")
-            threading.Timer(
-                0.8,
-                lambda: os.kill(os.getpid(), signal.SIGTERM),
-            ).start()
-            st.stop()
+            st.caption("顶部真实计数对应当前 config 指向的数据库。")
+            task_dir_text = resource_root_text = ""
+            selected_identity = selected_source = ""
+            load_button = False
+
+        _render_shutdown(service_bundle)
 
     if mode == "Mock Demo":
         st.subheader("Mock Demo")
-        if uploaded_query is not None:
-            st.image(uploaded_query, caption=f"仅内存预览：{uploaded_query.name}", width=420)
-            st.caption("本次上传未保存到磁盘。")
+        uploaded_preview = st.file_uploader(
+            "可选：仅在内存中预览查询图片",
+            type=["jpg", "jpeg", "png", "webp"],
+            accept_multiple_files=False,
+            help="该图片不会写入源码目录。",
+            key="mock_preview",
+        )
+        if uploaded_preview is not None:
+            st.image(
+                uploaded_preview,
+                caption=f"仅内存预览：{uploaded_preview.name}",
+                width=420,
+            )
         else:
-            st.info("无需上传即可加载内置 Mock；可选上传只做内存预览。")
-
+            st.info("无需上传即可加载内置 Mock。")
         if load_button:
-            if not identity_libraries or not source_libraries:
-                st.error("内置演示库不完整。")
-            else:
-                try:
-                    identity_path = next(
-                        path for path in identity_libraries if path.name == selected_identity
-                    )
-                    source_path = next(
-                        path for path in source_libraries if path.name == selected_source
-                    )
-                    _load_mock(identity_path, source_path)
-                except (ResultDataError, StopIteration) as exc:
-                    st.error(f"Mock 结果拒绝加载：{exc}")
+            try:
+                identity_path = next(
+                    path for path in identity_libraries if path.name == selected_identity
+                )
+                source_path = next(
+                    path for path in source_libraries if path.name == selected_source
+                )
+                _load_mock(identity_path, source_path)
+            except (ResultDataError, StopIteration) as exc:
+                st.error(f"Mock 结果拒绝加载：{exc}")
+        _render_loaded_results("Mock Demo")
     elif mode == "Task Result":
         st.subheader("Task Result")
         st.info(
-            "只读 task_dir/identity_result.json 与 source_result.json；"
-            "task_dir 和批准资源根之外的路径会被拒绝。"
+            "只读 task_dir 下固定命名的三份结果 JSON；"
+            "旧双模块 task 如缺少 detection_result.json，AIGC 会显式标记为 skipped。"
         )
         if load_button:
             try:
@@ -332,43 +803,15 @@
                 )
             except (ResultDataError, OSError, RuntimeError) as exc:
                 st.error(f"Task Result 拒绝加载：{exc}")
+        _render_loaded_results("Task Result")
     else:
-        st.subheader("Live Analysis")
-        st.info(
-            "一次提交创建一个独立 task_dir；身份搜索和底图搜索顺序运行、互不阻断。"
+        _render_live_mode(
+            modules=modules,
+            service_bundle=service_bundle,
+            identity_root_text=identity_root_text,
+            source_root_text=source_root_text,
+            live_config_error=live_config_error,
         )
-        if uploaded_query is not None:
-            st.image(uploaded_query, caption=f"待分析：{uploaded_query.name}", width=420)
-        if load_button:
-            try:
-                _run_live_analysis(
-                    uploaded_query,
-                    identity_root_text,
-                    source_root_text,
-                )
-            except (ResultDataError, OrchestrationError, OSError, RuntimeError) as exc:
-                st.error(f"Live Analysis 失败：{exc}")
-        live_summary = st.session_state.get("live_task_status")
-        loaded = st.session_state.get("loaded_results")
-        if live_summary and loaded and loaded.get("mode") == "Live Analysis":
-            left, middle, right = st.columns(3)
-            left.metric("总状态", live_summary["status"])
-            middle.metric(
-                "身份模块",
-                live_summary["identity"]["status"],
-                f"{live_summary['identity']['runtime_seconds']:.3f} 秒",
-                delta_color="off",
-            )
-            right.metric(
-                "底图模块",
-                live_summary["source"]["status"],
-                f"{live_summary['source']['runtime_seconds']:.3f} 秒",
-                delta_color="off",
-            )
-            st.write(f"**task_id：** `{live_summary['task_id']}`")
-            st.write(f"**总耗时：** `{live_summary['runtime_seconds']:.3f} 秒`")
-
-    _render_loaded_results(mode)
 
 
 if __name__ == "__main__":
```

### `frontend/components/data_loader.py`

```diff
--- base/frontend/components/data_loader.py
+++ target/frontend/components/data_loader.py
@@ -10,6 +10,7 @@
 
 
 ResultKind = Literal["identity", "source"]
+ResultStatus = Literal["success", "error", "skipped"]
 MAX_RESULT_BYTES = 5 * 1024 * 1024
 
 COMMON_FIELDS = {
@@ -34,6 +35,16 @@
     "final_score",
     "visualization_path",
 }
+DETECTION_FIELDS = {
+    "status",
+    "query_image",
+    "is_deepfake",
+    "score",
+    "label",
+    "runtime_seconds",
+    "details",
+    "error_message",
+}
 
 
 class ResultDataError(ValueError):
@@ -43,9 +54,11 @@
 @dataclass(frozen=True)
 class TaskResults:
     task_dir: Path
+    deepfake_result: dict[str, Any]
     identity_result: dict[str, Any]
     source_result: dict[str, Any]
     path_policy: ResourcePathPolicy
+    legacy_dual_module: bool = False
 
 
 def _require_nonempty_string(value: Any, field: str) -> str:
@@ -101,8 +114,8 @@
         raise ResultDataError(f"JSON 缺少字段：{'、'.join(sorted(missing))}")
 
     status = data["status"]
-    if status not in {"success", "error"}:
-        raise ResultDataError("status 只允许 'success' 或 'error'。")
+    if status not in {"success", "error", "skipped"}:
+        raise ResultDataError("status 只允许 'success'、'error' 或 'skipped'。")
     query_image = _require_nonempty_string(data["query_image"], "query_image")
     if not isinstance(data["candidates"], list):
         raise ResultDataError("candidates 必须是数组。")
@@ -115,6 +128,8 @@
         raise ResultDataError("status=error 时 error_message 必须是非空字符串。")
     if status == "error" and data["candidates"]:
         raise ResultDataError("status=error 时 candidates 必须为空数组。")
+    if status == "skipped" and data["candidates"]:
+        raise ResultDataError("status=skipped 时 candidates 必须为空数组。")
 
     try:
         path_policy.resolve(query_image, "query", label="query_image")
@@ -234,6 +249,91 @@
     return load_result(json_path, "source", path_policy)
 
 
+def validate_detection_data(
+    data: dict[str, Any],
+    path_policy: ResourcePathPolicy,
+) -> dict[str, Any]:
+    """Validate the AIGC/deepfake result without changing native uncertainty.
+
+    ``LOW_CONFIDENCE`` and ``INSUFFICIENT_EVIDENCE`` are represented by a
+    successful module call with ``is_deepfake=null``.  They intentionally stay
+    uncertain here; the loader never maps them to ``real``.
+    """
+
+    if not isinstance(data, dict):
+        raise ResultDataError("鉴伪 JSON 顶层必须是对象。")
+    missing = DETECTION_FIELDS - data.keys()
+    if missing:
+        raise ResultDataError(f"鉴伪 JSON 缺少字段：{'、'.join(sorted(missing))}")
+
+    status = data["status"]
+    if status not in {"success", "error", "skipped"}:
+        raise ResultDataError("鉴伪 status 只允许 'success'、'error' 或 'skipped'。")
+
+    query_image = _require_nonempty_string(data["query_image"], "query_image")
+    try:
+        path_policy.resolve(query_image, "query", label="query_image")
+    except PathContainmentError as exc:
+        raise ResultDataError(str(exc)) from exc
+
+    is_deepfake = data["is_deepfake"]
+    if is_deepfake is not None and not isinstance(is_deepfake, bool):
+        raise ResultDataError("is_deepfake 必须是布尔值或 null。")
+
+    score = data["score"]
+    if score is not None:
+        number = _require_number(score, "score")
+        if not 0.0 <= number <= 1.0:
+            raise ResultDataError("score 必须在 0 到 1 之间。")
+
+    label = data["label"]
+    if label is not None and not isinstance(label, str):
+        raise ResultDataError("label 必须是字符串或 null。")
+    _require_number(data["runtime_seconds"], "runtime_seconds", nonnegative=True)
+    if not isinstance(data["details"], dict):
+        raise ResultDataError("details 必须是对象。")
+
+    error_message = data["error_message"]
+    if error_message is not None and not isinstance(error_message, str):
+        raise ResultDataError("鉴伪 error_message 必须是字符串或 null。")
+    if status == "success" and error_message is not None:
+        raise ResultDataError("鉴伪 status=success 时 error_message 必须为 null。")
+    if status == "error" and (
+        not isinstance(error_message, str) or not error_message.strip()
+    ):
+        raise ResultDataError("鉴伪 status=error 时 error_message 必须是非空字符串。")
+    if status == "skipped":
+        if is_deepfake is not None or score is not None:
+            raise ResultDataError(
+                "鉴伪 status=skipped 时 is_deepfake 和 score 必须为 null。"
+            )
+        if label not in {None, "skipped"}:
+            raise ResultDataError("鉴伪 status=skipped 时 label 只能为 null 或 'skipped'。")
+    return data
+
+
+def load_detection_result(
+    json_path: Path,
+    path_policy: ResourcePathPolicy,
+) -> dict[str, Any]:
+    return validate_detection_data(_read_json(json_path), path_policy)
+
+
+def skipped_detection_result(query_image: str, reason: str) -> dict[str, Any]:
+    """Create the explicit compatibility marker for an old two-module task."""
+
+    return {
+        "status": "skipped",
+        "query_image": query_image,
+        "is_deepfake": None,
+        "score": None,
+        "label": "skipped",
+        "runtime_seconds": 0.0,
+        "details": {"reason": reason, "compatibility_mode": "legacy_dual_module"},
+        "error_message": None,
+    }
+
+
 def load_task_results(
     task_dir: Path,
     *,
@@ -253,4 +353,22 @@
     )
     identity = load_identity_result(task_path / "identity_result.json", policy)
     source = load_source_result(task_path / "source_result.json", policy)
-    return TaskResults(task_path, identity, source, policy)
+    detection_path = task_path / "detection_result.json"
+    legacy_dual_module = not detection_path.is_file()
+    if legacy_dual_module:
+        query_image = str(identity.get("query_image") or source.get("query_image") or "")
+        deepfake = skipped_detection_result(
+            query_image,
+            "旧版双模块 task 未包含 detection_result.json。",
+        )
+        validate_detection_data(deepfake, policy)
+    else:
+        deepfake = load_detection_result(detection_path, policy)
+    return TaskResults(
+        task_path,
+        deepfake,
+        identity,
+        source,
+        policy,
+        legacy_dual_module,
+    )
```

### `frontend/components/result_views.py`

```diff
--- base/frontend/components/result_views.py
+++ target/frontend/components/result_views.py
@@ -8,8 +8,201 @@
 from components.path_security import ResourcePathPolicy
 
 
+UNCERTAIN_DEEPFAKE_LABELS = {
+    "low_confidence",
+    "insufficient",
+    "insufficient_evidence",
+    "undetermined",
+    "unknown",
+}
+
+
 def format_score(value: Any) -> str:
     return f"{float(value):.4f}"
+
+
+def _optional_score(value: Any) -> str:
+    return format_score(value) if value is not None else "—"
+
+
+def _skip_reason(data: dict[str, Any]) -> str:
+    details = data.get("details")
+    if isinstance(details, dict) and details.get("reason"):
+        return str(details["reason"])
+    return str(data.get("error_message") or "本次任务未选择该模块。")
+
+
+def _deepfake_evidence(data: dict[str, Any]) -> dict[str, str]:
+    status = str(data.get("status") or "error").lower()
+    if status == "skipped":
+        return {
+            "title": "AIGC 鉴伪",
+            "state": "info",
+            "headline": "未运行",
+            "detail": _skip_reason(data),
+        }
+    if status != "success":
+        return {
+            "title": "AIGC 鉴伪",
+            "state": "error",
+            "headline": "执行失败",
+            "detail": str(data.get("error_message") or "未知错误"),
+        }
+
+    label = str(data.get("label") or "undetermined").strip().lower().replace("-", "_")
+    score_text = (
+        f"FAKE 风险 score={format_score(data['score'])}"
+        if data.get("score") is not None
+        else "未返回 score"
+    )
+    if data.get("is_deepfake") is None or label in UNCERTAIN_DEEPFAKE_LABELS:
+        details = data.get("details") or {}
+        native_status = str(details.get("native_status") or label).upper()
+        return {
+            "title": "AIGC 鉴伪",
+            "state": "warning",
+            "headline": f"不确定（{native_status}）",
+            "detail": f"{score_text}；证据不足，不能解释为 REAL。",
+        }
+    if label not in {"deepfake", "real"}:
+        return {
+            "title": "AIGC 鉴伪",
+            "state": "warning",
+            "headline": f"不确定（{label or 'UNKNOWN'}）",
+            "detail": f"{score_text}；未知标签不会被映射为 REAL。",
+        }
+    if (label == "deepfake" and data.get("is_deepfake") is False) or (
+        label == "real" and data.get("is_deepfake") is True
+    ):
+        return {
+            "title": "AIGC 鉴伪",
+            "state": "warning",
+            "headline": "不确定（返回字段不一致）",
+            "detail": f"{score_text}；冲突结果不会被强行解释为 REAL/FAKE。",
+        }
+    if data.get("is_deepfake") is True or label == "deepfake":
+        return {
+            "title": "AIGC 鉴伪",
+            "state": "warning",
+            "headline": "疑似 AI 换脸",
+            "detail": score_text,
+        }
+    if data.get("is_deepfake") is False or label == "real":
+        return {
+            "title": "AIGC 鉴伪",
+            "state": "success",
+            "headline": "未检出 AI 换脸特征",
+            "detail": f"{score_text}；这不等同于司法真实性结论。",
+        }
+    return {
+        "title": "AIGC 鉴伪",
+        "state": "warning",
+        "headline": "不确定（标签与布尔结论不一致）",
+        "detail": f"{score_text}；冲突结果不会被强行解释为 REAL/FAKE。",
+    }
+
+
+def _identity_evidence(data: dict[str, Any]) -> dict[str, str]:
+    status = str(data.get("status") or "error").lower()
+    if status == "skipped":
+        return {
+            "title": "人脸身份搜索",
+            "state": "info",
+            "headline": "未运行",
+            "detail": _skip_reason(data),
+        }
+    if status != "success":
+        return {
+            "title": "人脸身份搜索",
+            "state": "error",
+            "headline": "执行失败",
+            "detail": str(data.get("error_message") or "未知错误"),
+        }
+    candidates = data.get("candidates") or []
+    if not candidates:
+        return {
+            "title": "人脸身份搜索",
+            "state": "warning",
+            "headline": "未返回身份候选",
+            "detail": "本次检索没有可用 Top1 线索。",
+        }
+    top = candidates[0]
+    return {
+        "title": "人脸身份搜索",
+        "state": "success",
+        "headline": f"Top1：{top.get('identity_name') or top.get('identity_id') or '未命名人员'}",
+        "detail": f"身份相似度={_optional_score(top.get('score'))}；仅作为候选线索。",
+    }
+
+
+def _source_evidence(data: dict[str, Any]) -> dict[str, str]:
+    status = str(data.get("status") or "error").lower()
+    if status == "skipped":
+        return {
+            "title": "原始底图搜索",
+            "state": "info",
+            "headline": "未运行",
+            "detail": _skip_reason(data),
+        }
+    if status != "success":
+        return {
+            "title": "原始底图搜索",
+            "state": "error",
+            "headline": "执行失败",
+            "detail": str(data.get("error_message") or "未知错误"),
+        }
+    candidates = data.get("candidates") or []
+    if not candidates:
+        return {
+            "title": "原始底图搜索",
+            "state": "warning",
+            "headline": "未返回底图候选",
+            "detail": "本次检索没有可用 Top1 线索。",
+        }
+    top = candidates[0]
+    display_name = Path(str(top.get("file_path") or "未命名文件")).name
+    return {
+        "title": "原始底图搜索",
+        "state": "success",
+        "headline": f"Top1：{display_name}",
+        "detail": f"底图最终分数={_optional_score(top.get('final_score'))}；仅作为候选线索。",
+    }
+
+
+def build_evidence_summary(
+    deepfake: dict[str, Any],
+    identity: dict[str, Any],
+    source: dict[str, Any],
+) -> dict[str, dict[str, str]]:
+    """Build a three-source evidence summary without a fused probability."""
+
+    return {
+        "deepfake": _deepfake_evidence(deepfake),
+        "identity": _identity_evidence(identity),
+        "source": _source_evidence(source),
+    }
+
+
+def render_evidence_summary(
+    deepfake: dict[str, Any],
+    identity: dict[str, Any],
+    source: dict[str, Any],
+) -> None:
+    st.subheader("证据汇总")
+    st.info(
+        "三个模块解决不同问题：本区只并列汇总证据，"
+        "不计算、不暗示任何跨模块“综合概率”。"
+    )
+    summary = build_evidence_summary(deepfake, identity, source)
+    columns = st.columns(3)
+    for column, key in zip(columns, ("deepfake", "identity", "source"), strict=True):
+        item = summary[key]
+        with column:
+            with st.container(border=True):
+                st.markdown(f"**{item['title']}**")
+                renderer = getattr(st, item["state"], st.info)
+                renderer(item["headline"])
+                st.caption(item["detail"])
 
 
 def identity_view_items(
@@ -79,12 +272,58 @@
         st.write(f"**模块耗时：** `{format_score(data['runtime_seconds'])} 秒`")
 
 
+def render_deepfake_result(data: dict[str, Any]) -> None:
+    st.subheader("AI 换脸鉴伪结果")
+    render_result_header(data)
+    if data["status"] == "skipped":
+        st.info(f"本次未运行 AI 换脸鉴伪：{_skip_reason(data)}")
+        return
+    if data["status"] != "success":
+        st.error(data["error_message"])
+        return
+
+    evidence = _deepfake_evidence(data)
+    getattr(st, evidence["state"], st.warning)(evidence["headline"])
+    st.caption(evidence["detail"])
+
+    details = data["details"]
+    verdict_column, score_column, device_column = st.columns(3)
+    verdict_column.metric(
+        "模块结论",
+        "疑似换脸"
+        if data["is_deepfake"] is True
+        else "未检出"
+        if data["is_deepfake"] is False
+        else "不确定",
+    )
+    score_column.metric("FAKE 风险 score", _optional_score(data["score"]))
+    device_column.metric("运行设备", str(details.get("actual_device") or "—"))
+
+    native_status = details.get("native_status")
+    if native_status:
+        st.write(f"**原生状态：** `{native_status}`")
+    threshold = details.get("decision_threshold")
+    if threshold is not None:
+        st.write(f"**决策阈值：** `{format_score(threshold)}`")
+    semantics = details.get("score_semantics")
+    if semantics:
+        st.caption(str(semantics))
+    disclaimer = details.get("disclaimer")
+    if disclaimer:
+        st.caption(str(disclaimer))
+    with st.expander("鉴伪运行详情与模型追溯信息"):
+        st.json(details)
+
+
 def render_identity_results(
     data: dict[str, Any],
     path_policy: ResourcePathPolicy,
 ) -> None:
     st.subheader("人脸身份搜索结果")
     render_result_header(data)
+    if data["status"] == "skipped":
+        st.info("本次任务未运行人脸身份搜索。")
+        return
     if data["status"] != "success":
         st.error(data["error_message"])
         return
@@ -117,6 +356,9 @@
 ) -> None:
     st.subheader("原始底图搜索结果")
     render_result_header(data)
+    if data["status"] == "skipped":
+        st.info("本次任务未运行原始底图搜索。")
+        return
     if data["status"] != "success":
         st.error(data["error_message"])
         return
@@ -150,4 +392,3 @@
             st.code(item["file_path"])
             st.write("**匹配图路径（结果原值）：**")
             st.code(item["visualization_path"])
-
```

### `frontend/mock_task/README.md`

```diff
--- base/frontend/mock_task/README.md
+++ target/frontend/mock_task/README.md
@@ -1,5 +1,4 @@
 # Mock Task
 
-这是一个符合任务目录读取模式的前端夹具，不包含任何 A/B 推理结果。
+这是一个符合三模块任务目录读取模式的前端夹具，不包含任何 AIGC/身份/底图真实推理结果。
 `query.svg` 与 `visualizations/` 位于 task 内；身份参考图和底图通过界面中显式批准的示例库根解析。
-
```

### `launcher.py`

```diff
--- base/launcher.py
+++ target/launcher.py
@@ -4,16 +4,23 @@
 import sys
 from pathlib import Path
 
-from orchestrator import DEFAULT_CONFIG, OrchestrationError, run_task
+from orchestrator import DEFAULT_CONFIG, MODULE_ORDER, OrchestrationError, run_task
 
 
 def build_parser() -> argparse.ArgumentParser:
     parser = argparse.ArgumentParser(
-        description="运行一次独立的身份搜索 + 原始底图搜索集成任务。"
+        description="运行一次鉴伪 + 身份搜索 + 原始底图搜索三分支集成任务。"
     )
     parser.add_argument("--query", required=True, help="查询图片路径")
     parser.add_argument("--identity-library", help="可选：覆盖身份库路径")
     parser.add_argument("--source-library", help="可选：覆盖底图库路径")
+    parser.add_argument(
+        "--modules",
+        nargs="+",
+        choices=MODULE_ORDER,
+        metavar="MODULE",
+        help="可选：仅运行指定模块；默认运行 deepfake identity source",
+    )
     parser.add_argument(
         "--config",
         default=str(DEFAULT_CONFIG),
@@ -29,6 +36,7 @@
             args.query,
             identity_library=args.identity_library,
             source_library=args.source_library,
+            selected_modules=args.modules,
             config_path=args.config,
             progress_callback=lambda message: print(f"[进度] {message}", flush=True),
         )
@@ -39,8 +47,11 @@
     print(f"task_id: {status['task_id']}")
     print(f"task_dir: {status['task_dir']}")
     print(f"status: {status['status']}")
+    print(f"selected_modules: {','.join(status['selected_modules'])}")
+    print(f"deepfake_status: {status['deepfake']['status']}")
     print(f"identity_status: {status['identity']['status']}")
     print(f"source_status: {status['source']['status']}")
+    print(f"deepfake_runtime: {status['deepfake']['runtime_seconds']:.6f}")
     print(f"identity_runtime: {status['identity']['runtime_seconds']:.6f}")
     print(f"source_runtime: {status['source']['runtime_seconds']:.6f}")
     print(f"total_runtime: {status['runtime_seconds']:.6f}")
```

### `orchestrator.py`

```diff
--- base/orchestrator.py
+++ target/orchestrator.py
@@ -1,15 +1,19 @@
 from __future__ import annotations
 
 import json
+import math
 import os
+import queue
 import secrets
 import shutil
 import subprocess
 import sys
+import threading
 import time
+from collections.abc import Collection, Mapping
 from datetime import datetime, timezone
 from pathlib import Path
-from typing import Any, Callable
+from typing import Any, Callable, Protocol
 
 
 PRODUCT_ROOT = Path(__file__).resolve().parent
@@ -17,6 +21,32 @@
     os.environ.get("FSTS_CONFIG_PATH", str(PRODUCT_ROOT / "config.json"))
 ).expanduser().resolve(strict=False)
 ProgressCallback = Callable[[str], None]
+MODULE_ORDER = ("deepfake", "identity", "source")
+SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
+
+
+class RuntimeServices(Protocol):
+    """Optional persistent-worker interface accepted by :func:`run_task`.
+
+    A service manager may implement either method independently. Missing
+    methods or infrastructure errors transparently use the subprocess adapter.
+    """
+
+    def run_deepfake(
+        self,
+        query_path: str | Path,
+        output_path: str | Path,
+        timeout: float,
+    ) -> dict[str, Any]: ...
+
+    def run_identity(
+        self,
+        query_path: str | Path,
+        output_path: str | Path,
+        library_path: str | Path,
+        top_k: int,
+        timeout: float,
+    ) -> dict[str, Any]: ...
 
 
 class OrchestrationError(RuntimeError):
@@ -44,6 +74,29 @@
         text = value
     path.parent.mkdir(parents=True, exist_ok=True)
     path.write_text(text, encoding="utf-8")
+
+
+class _ProgressReporter:
+    def __init__(self, callback: ProgressCallback | None, log_path: Path) -> None:
+        self._callback = callback or (lambda _message: None)
+        self._log_path = log_path
+        self.events: list[dict[str, Any]] = []
+
+    def emit(self, stage: str, message: str, module: str | None = None) -> None:
+        event = {
+            "timestamp": _now_iso(),
+            "stage": stage,
+            "module": module,
+            "message": message,
+        }
+        self.events.append(event)
+        _write_json(self._log_path, self.events)
+        try:
+            self._callback(message)
+        except Exception as exc:
+            # A display callback is not allowed to terminate forensic work.
+            event["callback_error"] = f"{type(exc).__name__}: {exc}"
+            _write_json(self._log_path, self.events)
 
 
 def load_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], Path]:
@@ -56,7 +109,7 @@
         raise OrchestrationError(f"无法读取配置文件：{path}；{exc}") from exc
     if not isinstance(config, dict):
         raise OrchestrationError("配置文件顶层必须是 JSON 对象。")
-    for section in ("runtime", "identity", "source", "frontend"):
+    for section in ("runtime", "deepfake", "identity", "source", "frontend"):
         if not isinstance(config.get(section), dict):
             raise OrchestrationError(f"配置缺少对象：{section}")
     return config, path
@@ -96,223 +149,244 @@
     raise OrchestrationError("无法生成唯一 task_id。")
 
 
-def _standard_error_result(query_path: Path, runtime: float, message: str) -> dict[str, Any]:
+def _standard_search_result(
+    query_path: Path,
+    runtime: float,
+    message: str | None,
+    *,
+    status: str = "error",
+) -> dict[str, Any]:
     return {
-        "status": "error",
+        "status": status,
         "query_image": str(query_path),
         "candidates": [],
         "runtime_seconds": max(0.0, float(runtime)),
-        "error_message": message.strip() or "未知模块错误",
+        "error_message": (
+            (message or "未知模块错误").strip() if status == "error" else None
+        ),
+        **(
+            {"details": {"reason": message or "未选择此模块"}}
+            if status == "skipped"
+            else {}
+        ),
     }
 
 
-def _validate_result(payload: Any) -> tuple[bool, str]:
+def _standard_deepfake_result(
+    query_path: Path,
+    runtime: float,
+    message: str | None,
+    *,
+    status: str = "error",
+) -> dict[str, Any]:
+    return {
+        "status": status,
+        "query_image": str(query_path),
+        "is_deepfake": None,
+        "score": None,
+        "label": "skipped" if status == "skipped" else None,
+        "runtime_seconds": max(0.0, float(runtime)),
+        "details": ({"reason": message or "未选择此模块"} if status == "skipped" else {}),
+        "error_message": (
+            (message or "未知鉴伪模块错误").strip() if status == "error" else None
+        ),
+    }
+
+
+def _validate_search_result(payload: Any) -> tuple[bool, str]:
     if not isinstance(payload, dict):
         return False, "结果 JSON 顶层不是对象"
     required = {"status", "query_image", "candidates", "runtime_seconds", "error_message"}
     missing = sorted(required - payload.keys())
     if missing:
         return False, f"结果 JSON 缺少字段：{', '.join(missing)}"
-    if payload["status"] not in {"success", "error"}:
+    if payload["status"] not in {"success", "error", "skipped"}:
         return False, f"未知结果状态：{payload['status']!r}"
     if not isinstance(payload["candidates"], list):
         return False, "结果 candidates 不是数组"
+    runtime = payload["runtime_seconds"]
+    if (
+        isinstance(runtime, bool)
+        or not isinstance(runtime, (int, float))
+        or not math.isfinite(float(runtime))
+        or float(runtime) < 0
+    ):
+        return False, "结果 runtime_seconds 必须为非负有限数值"
+    if payload["status"] == "success" and payload["error_message"] is not None:
+        return False, "成功结果的 error_message 必须为 null"
+    if payload["status"] == "error" and not str(payload["error_message"] or "").strip():
+        return False, "失败结果必须包含非空 error_message"
     return True, ""
 
 
-def _run_module(
+def _validate_deepfake_result(payload: Any) -> tuple[bool, str]:
+    if not isinstance(payload, dict):
+        return False, "鉴伪结果 JSON 顶层不是对象"
+    required = {
+        "status",
+        "query_image",
+        "is_deepfake",
+        "score",
+        "label",
+        "runtime_seconds",
+        "details",
+        "error_message",
+    }
+    missing = sorted(required - payload.keys())
+    if missing:
+        return False, f"鉴伪结果 JSON 缺少字段：{', '.join(missing)}"
+    if payload["status"] not in {"success", "error", "skipped"}:
+        return False, f"未知鉴伪结果状态：{payload['status']!r}"
+    if payload["is_deepfake"] is not None and not isinstance(payload["is_deepfake"], bool):
+        return False, "鉴伪结果 is_deepfake 必须为布尔值或 null"
+    score = payload["score"]
+    if score is not None:
+        if isinstance(score, bool) or not isinstance(score, (int, float)):
+            return False, "鉴伪结果 score 必须为数值或 null"
+        if not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0:
+            return False, "鉴伪结果 score 必须是 0 到 1 之间的有限数值"
+    if payload["label"] is not None and not isinstance(payload["label"], str):
+        return False, "鉴伪结果 label 必须为字符串或 null"
+    if not isinstance(payload["details"], dict):
+        return False, "鉴伪结果 details 必须为对象"
+    runtime = payload["runtime_seconds"]
+    if (
+        isinstance(runtime, bool)
+        or not isinstance(runtime, (int, float))
+        or not math.isfinite(float(runtime))
+        or float(runtime) < 0
+    ):
+        return False, "鉴伪结果 runtime_seconds 必须为非负有限数值"
+    if payload["status"] == "success" and payload["error_message"] is not None:
+        return False, "鉴伪成功结果的 error_message 必须为 null"
+    if payload["status"] == "error" and not str(payload["error_message"] or "").strip():
+        return False, "鉴伪失败结果必须包含非空 error_message"
+    return True, ""
+
+
+def _service_method(services: Any, name: str) -> Callable[..., Any] | None:
+    if services is None:
+        return None
+    candidate = services.get(name) if isinstance(services, Mapping) else getattr(services, name, None)
+    return candidate if callable(candidate) else None
+
+
+def _invoke_service(
+    method: Callable[..., Any],
+    *,
+    kwargs: dict[str, Any],
+    timeout: float,
+) -> Any:
+    """Bound an in-process service call without making its model non-persistent."""
+
+    outcomes: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)
+
+    def target() -> None:
+        try:
+            outcomes.put((True, method(**kwargs)))
+        except BaseException as exc:  # transported to the orchestration thread
+            outcomes.put((False, exc))
+
+    thread = threading.Thread(target=target, name="fsts-service-call", daemon=True)
+    thread.start()
+    try:
+        # ``timeout`` is the caller's remaining module budget, not a fresh
+        # per-attempt allowance.  The outer guard must not extend that budget.
+        ok, value = outcomes.get(timeout=timeout)
+    except queue.Empty as exc:
+        raise TimeoutError(f"持久服务调用超时（{timeout:g} 秒）") from exc
+    if ok:
+        return value
+    if not isinstance(value, Exception):
+        raise RuntimeError(f"持久服务异常退出：{type(value).__name__}") from value
+    raise value
+
+
+def _remaining_timeout(deadline: float, *, module: str) -> float:
+    """Return the positive remainder of one module-wide monotonic deadline."""
+
+    remaining = deadline - time.monotonic()
+    if remaining <= 0:
+        raise TimeoutError(f"{module} 已耗尽模块总超时预算")
+    return remaining
+
+
+def _run_subprocess(
+    command: list[str],
+    *,
+    cwd: Path,
+    timeout: float,
+    hide_console: bool,
+    executable_worker: bool,
+    extra_environment: Mapping[str, str] | None = None,
+) -> subprocess.CompletedProcess[str]:
+    child_environment = os.environ.copy()
+    if extra_environment:
+        child_environment.update(extra_environment)
+    restore_dll_directory: str | None = None
+    if executable_worker:
+        # Frozen workers must not inherit the front-end bundle's ABI/DLL state.
+        child_environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
+        if os.name == "nt" and getattr(sys, "frozen", False):
+            import ctypes
+
+            restore_dll_directory = str(getattr(sys, "_MEIPASS", "")) or None
+            ctypes.windll.kernel32.SetDllDirectoryW(None)
+    try:
+        return subprocess.run(
+            command,
+            cwd=cwd,
+            capture_output=True,
+            text=True,
+            encoding="utf-8",
+            errors="replace",
+            timeout=timeout,
+            check=False,
+            env=child_environment,
+            creationflags=(
+                subprocess.CREATE_NO_WINDOW
+                if os.name == "nt" and hide_console
+                else 0
+            ),
+        )
+    finally:
+        if restore_dll_directory is not None:
+            import ctypes
+
+            ctypes.windll.kernel32.SetDllDirectoryW(restore_dll_directory)
+
+
+def _finish_module(
     *,
     name: str,
-    section: dict[str, Any],
-    config_path: Path,
-    query_path: Path,
-    library_override: str | Path | None,
+    payload: dict[str, Any],
     output_path: Path,
     logs_dir: Path,
-    identity_cache: Path,
-    progress: ProgressCallback,
+    started_at: str,
+    started: float,
+    returncode: int | None,
+    timed_out: bool,
+    stdout: str = "",
+    stderr: str = "",
+    command_record: dict[str, Any] | None = None,
+    execution_mode: str,
+    fallback_reason: str | None = None,
+    skipped: bool = False,
 ) -> tuple[dict[str, Any], dict[str, Any]]:
-    started_at = _now_iso()
-    started = time.perf_counter()
+    runtime = max(0.0, time.perf_counter() - started)
+    status = str(payload.get("status", "error"))
+    _write_json(output_path, payload)
     stdout_path = logs_dir / f"{name}_stdout.log"
     stderr_path = logs_dir / f"{name}_stderr.log"
+    command_path = logs_dir / f"{name}_command.json"
     module_log_path = logs_dir / f"{name}.log"
-    command_path = logs_dir / f"{name}_command.json"
-
-    try:
-        executable_value = section.get("executable")
-        if executable_value:
-            executable_path = resolve_path(executable_value, config_path)
-            python_path = None
-            runner_path = None
-            command_prefix = [str(executable_path)]
-            module_cwd = executable_path.parent
-        else:
-            python_path = resolve_path(section["python"], config_path)
-            runner_path = resolve_path(section["runner"], config_path)
-            executable_path = None
-            command_prefix = [str(python_path), "-B", str(runner_path)]
-            module_cwd = runner_path.parent
-        library_path = resolve_path(library_override or section["library"], config_path)
-        timeout = float(section.get("timeout_seconds", 600))
-        top_k = int(section.get("top_k", 5))
-    except (KeyError, TypeError, ValueError) as exc:
-        message = f"{name} 配置无效：{exc}"
-        result = _standard_error_result(query_path, time.perf_counter() - started, message)
-        _write_json(output_path, result)
-        _write_text(stdout_path, "")
-        _write_text(stderr_path, message + "\n")
-        metadata = {
-            "module": name,
-            "status": "error",
-            "started_at": started_at,
-            "finished_at": _now_iso(),
-            "runtime_seconds": result["runtime_seconds"],
-            "returncode": None,
-            "error": message,
-        }
-        _write_json(module_log_path.with_suffix(".json"), metadata)
-        _write_text(module_log_path, message + "\n")
-        return result, metadata
-
-    errors: list[str] = []
-    if executable_path is not None:
-        if not executable_path.is_file():
-            errors.append(f"模块可执行文件不存在：{executable_path}")
-    else:
-        if python_path is None or not python_path.is_file():
-            errors.append(f"Python 不存在：{python_path}")
-        if runner_path is None or not runner_path.is_file():
-            errors.append(f"runner 不存在：{runner_path}")
-    if not library_path.is_dir():
-        errors.append(f"图库不存在或不是目录：{library_path}")
-    if top_k < 1:
-        errors.append(f"top_k 必须大于 0：{top_k}")
-    if timeout <= 0:
-        errors.append(f"timeout_seconds 必须大于 0：{timeout}")
-
-    command = command_prefix + [
-        "--query",
-        str(query_path),
-        "--library",
-        str(library_path),
-        "--top-k",
-        str(top_k),
-        "--output",
-        str(output_path),
-    ]
-    if name == "identity":
-        identity_cache.mkdir(parents=True, exist_ok=True)
-        command.extend(["--ctx", str(section.get("ctx", -1))])
-        command.extend(["--cache-dir", str(identity_cache)])
-        if section.get("model_root"):
-            command.extend(["--model-root", str(resolve_path(section["model_root"], config_path))])
-        if section.get("offline", True):
-            command.append("--offline")
-
+    _write_text(stdout_path, stdout)
+    _write_text(stderr_path, stderr)
     _write_json(
         command_path,
-        {
-            "module": name,
-            "argv": command,
-            "cwd": str(module_cwd),
-            "library": str(library_path),
-            "output": str(output_path),
-        },
+        command_record
+        or {"module": name, "mode": execution_mode, "argv": [], "output": str(output_path)},
     )
-
-    returncode: int | None = None
-    timed_out = False
-    stdout = ""
-    stderr = ""
-    if errors:
-        stderr = "\n".join(errors) + "\n"
-    else:
-        progress(
-            "正在进行人脸身份搜索……"
-            if name == "identity"
-            else "正在进行原始底图搜索……"
-        )
-        try:
-            child_environment = os.environ.copy()
-            restore_dll_directory: str | None = None
-            if executable_path is not None:
-                # A child PyInstaller application must start a fresh runtime.  It
-                # must also not inherit the front-end bundle's DLL search path,
-                # because A and B intentionally use different NumPy/OpenCV ABIs.
-                child_environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
-                if os.name == "nt" and getattr(sys, "frozen", False):
-                    import ctypes
-
-                    restore_dll_directory = str(getattr(sys, "_MEIPASS", "")) or None
-                    ctypes.windll.kernel32.SetDllDirectoryW(None)
-            try:
-                completed = subprocess.run(
-                    command,
-                    cwd=module_cwd,
-                    capture_output=True,
-                    text=True,
-                    encoding="utf-8",
-                    errors="replace",
-                    timeout=timeout,
-                    check=False,
-                    env=child_environment,
-                    creationflags=(
-                        subprocess.CREATE_NO_WINDOW
-                        if os.name == "nt" and bool(section.get("hide_console", False))
-                        else 0
-                    ),
-                )
-            finally:
-                if restore_dll_directory is not None:
-                    ctypes.windll.kernel32.SetDllDirectoryW(restore_dll_directory)
-            returncode = completed.returncode
-            stdout = completed.stdout
-            stderr = completed.stderr
-        except subprocess.TimeoutExpired as exc:
-            timed_out = True
-            stdout = exc.stdout or ""
-            stderr = exc.stderr or ""
-            errors.append(f"{name} 超时（{timeout:g} 秒）")
-        except OSError as exc:
-            errors.append(f"无法启动 {name} 子进程：{exc}")
-            stderr = (stderr + "\n" + str(exc)).strip() + "\n"
-
-    _write_text(stdout_path, stdout)
-    _write_text(stderr_path, stderr)
-    runtime = time.perf_counter() - started
-
-    payload: Any = None
-    if output_path.is_file():
-        try:
-            payload = json.loads(output_path.read_text(encoding="utf-8"))
-            valid, validation_error = _validate_result(payload)
-            if not valid:
-                errors.append(validation_error)
-        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
-            errors.append(f"无法解析 {name} 结果 JSON：{exc}")
-    else:
-        errors.append(f"{name} 未生成结果文件")
-
-    if returncode not in (0, None):
-        errors.append(f"{name} 子进程退出码为 {returncode}")
-    if isinstance(payload, dict) and payload.get("status") == "error":
-        errors.append(str(payload.get("error_message") or f"{name} 返回 error"))
-
-    if errors:
-        if output_path.is_file():
-            raw_copy = logs_dir / f"{name}_raw_result.json"
-            try:
-                shutil.copy2(output_path, raw_copy)
-            except OSError:
-                pass
-        message = "; ".join(dict.fromkeys(error.strip() for error in errors if error.strip()))
-        payload = _standard_error_result(query_path, runtime, message)
-        _write_json(output_path, payload)
-        status = "error"
-    else:
-        status = "success"
-
     metadata = {
         "module": name,
         "status": status,
@@ -321,6 +395,9 @@
         "runtime_seconds": runtime,
         "returncode": returncode,
         "timed_out": timed_out,
+        "skipped": skipped or status == "skipped",
+        "execution_mode": execution_mode,
+        "fallback_reason": fallback_reason,
         "stdout_log": str(stdout_path),
         "stderr_log": str(stderr_path),
         "command_log": str(command_path),
@@ -338,13 +415,708 @@
                 f"runtime_seconds={runtime:.6f}",
                 f"returncode={returncode}",
                 f"timed_out={str(timed_out).lower()}",
+                f"skipped={str(metadata['skipped']).lower()}",
+                f"execution_mode={execution_mode}",
+                f"fallback_reason={fallback_reason or ''}",
                 f"error={metadata['error'] or ''}",
             ]
         )
         + "\n",
     )
-    progress(f"{name} 模块完成：{status}")
     return payload, metadata
+
+
+def _skip_module(
+    *,
+    name: str,
+    query_path: Path,
+    output_path: Path,
+    logs_dir: Path,
+    reason: str,
+) -> tuple[dict[str, Any], dict[str, Any]]:
+    started_at = _now_iso()
+    started = time.perf_counter()
+    payload = (
+        _standard_deepfake_result(query_path, 0.0, reason, status="skipped")
+        if name == "deepfake"
+        else _standard_search_result(query_path, 0.0, reason, status="skipped")
+    )
+    return _finish_module(
+        name=name,
+        payload=payload,
+        output_path=output_path,
+        logs_dir=logs_dir,
+        started_at=started_at,
+        started=started,
+        returncode=None,
+        timed_out=False,
+        command_record={
+            "module": name,
+            "mode": "skipped",
+            "argv": [],
+            "reason": reason,
+            "output": str(output_path),
+        },
+        execution_mode="skipped",
+        skipped=True,
+    )
+
+
+def _library_has_images(path: Path) -> bool:
+    try:
+        return any(
+            item.is_file() and item.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
+            for item in path.rglob("*")
+        )
+    except OSError:
+        return False
+
+
+def _run_deepfake_module(
+    *,
+    section: dict[str, Any],
+    config_path: Path,
+    query_path: Path,
+    output_path: Path,
+    logs_dir: Path,
+    progress: _ProgressReporter,
+    services: Any = None,
+) -> tuple[dict[str, Any], dict[str, Any]]:
+    name = "deepfake"
+    started_at = _now_iso()
+    started = time.perf_counter()
+    enabled = section.get("enabled", True)
+    if not isinstance(enabled, bool):
+        message = "deepfake 配置无效：enabled 必须为布尔值"
+        payload = _standard_deepfake_result(query_path, 0.0, message)
+        return _finish_module(
+            name=name,
+            payload=payload,
+            output_path=output_path,
+            logs_dir=logs_dir,
+            started_at=started_at,
+            started=started,
+            returncode=None,
+            timed_out=False,
+            stderr=message + "\n",
+            execution_mode="configuration",
+        )
+    if not enabled:
+        message = "鉴伪模块已由配置关闭，无法执行用户选择的模块"
+        payload = _standard_deepfake_result(query_path, 0.0, message)
+        return _finish_module(
+            name=name,
+            payload=payload,
+            output_path=output_path,
+            logs_dir=logs_dir,
+            started_at=started_at,
+            started=started,
+            returncode=None,
+            timed_out=False,
+            stderr=message + "\n",
+            command_record={
+                "module": name,
+                "mode": "disabled",
+                "argv": [],
+                "reason": message,
+                "output": str(output_path),
+            },
+            execution_mode="disabled",
+        )
+
+    try:
+        timeout = float(section.get("timeout_seconds", 180))
+    except (TypeError, ValueError) as exc:
+        timeout = 0.0
+        config_error = f"deepfake 配置无效：{exc}"
+    else:
+        config_error = (
+            ""
+            if math.isfinite(timeout) and timeout > 0
+            else f"timeout_seconds 必须是大于 0 的有限数值：{timeout}"
+        )
+    if config_error:
+        payload = _standard_deepfake_result(query_path, 0.0, config_error)
+        return _finish_module(
+            name=name,
+            payload=payload,
+            output_path=output_path,
+            logs_dir=logs_dir,
+            started_at=started_at,
+            started=started,
+            returncode=None,
+            timed_out=False,
+            stderr=config_error + "\n",
+            execution_mode="configuration",
+        )
+
+    deadline = time.monotonic() + timeout
+    progress.emit("loading_aigc", "正在加载AIGC模型……", name)
+    progress.emit("detecting_image", "正在检测图像……", name)
+    service_error: str | None = None
+    method = _service_method(services, "run_deepfake")
+    if method is not None:
+        # Keep the temporary JSON under the task root so the frozen adapter's
+        # native visualizations still land in task_dir/visualizations/deepfake.
+        # The distinct filename prevents a late service response racing the
+        # canonical subprocess result.
+        service_output_path = output_path.with_name(".deepfake_service_result.json")
+        try:
+            service_timeout = _remaining_timeout(deadline, module=name)
+            payload = _invoke_service(
+                method,
+                kwargs={
+                    "query_path": query_path,
+                    "output_path": service_output_path,
+                    "timeout": service_timeout,
+                },
+                timeout=service_timeout,
+            )
+            valid, validation_error = _validate_deepfake_result(payload)
+            if not valid:
+                raise ValueError(validation_error)
+            return _finish_module(
+                name=name,
+                payload=payload,
+                output_path=output_path,
+                logs_dir=logs_dir,
+                started_at=started_at,
+                started=started,
+                returncode=None,
+                timed_out=False,
+                command_record={
+                    "module": name,
+                    "mode": "service",
+                    "method": "run_deepfake",
+                    "timeout_seconds": timeout,
+                    "attempt_timeout_seconds": service_timeout,
+                    "service_output": str(service_output_path),
+                    "output": str(output_path),
+                },
+                execution_mode="service",
+            )
+        except Exception as exc:
+            service_error = f"{type(exc).__name__}: {exc}"
+            progress.emit("deepfake_service_fallback", "AIGC常驻服务不可用，切换隔离子进程……", name)
+
+    errors: list[str] = []
+    command: list[str] = []
+    python_path: Path | None = None
+    runner_path: Path | None = None
+    install_root: Path | None = None
+    torch_home: Path | None = None
+    base_weight: Path | None = None
+    device = "auto"
+    executable_worker = False
+    try:
+        executable_value = section.get("executable")
+        if executable_value:
+            executable_path = resolve_path(executable_value, config_path)
+            command_prefix = [str(executable_path)]
+            runner_path = executable_path
+            executable_worker = True
+        else:
+            python_path = resolve_path(section["python"], config_path)
+            runner_path = resolve_path(section["runner"], config_path)
+            python_args = section.get("python_args", ["-S", "-B"])
+            if not isinstance(python_args, list) or not all(isinstance(item, str) for item in python_args):
+                raise ValueError("python_args 必须为字符串数组")
+            command_prefix = [str(python_path), *python_args, str(runner_path)]
+        install_root = resolve_path(section["install_root"], config_path)
+        torch_home = resolve_path(section["torch_home"], config_path)
+        if section.get("base_weight"):
+            base_weight = resolve_path(section["base_weight"], config_path)
+        device = str(section.get("device", "auto"))
+        if device not in {"cpu", "auto", "cuda"}:
+            raise ValueError(f"未知 device：{device}")
+        extra_args = section.get("extra_args", [])
+        if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
+            raise ValueError("extra_args 必须为字符串数组")
+        command = [
+            *command_prefix,
+            "--query",
+            str(query_path),
+            "--output",
+            str(output_path),
+            "--install-root",
+            str(install_root),
+            "--torch-home",
+            str(torch_home),
+            "--device",
+            device,
+            *extra_args,
+        ]
+    except (KeyError, TypeError, ValueError) as exc:
+        errors.append(f"deepfake 配置无效：{exc}")
+
+    if executable_worker:
+        if runner_path is None or not runner_path.is_file():
+            errors.append(f"模块可执行文件不存在：{runner_path}")
+    else:
+        if python_path is None or not python_path.is_file():
+            errors.append(f"Python 不存在：{python_path}")
+        if runner_path is None or not runner_path.is_file():
+            errors.append(f"runner 不存在：{runner_path}")
+    if install_root is None or not install_root.is_dir():
+        errors.append(f"鉴伪安装目录不存在或不是目录：{install_root}")
+    if torch_home is not None:
+        try:
+            torch_home.mkdir(parents=True, exist_ok=True)
+        except OSError as exc:
+            errors.append(f"无法创建鉴伪隔离缓存目录：{exc}")
+    if base_weight is not None and not base_weight.is_file():
+        errors.append(f"MobileNetV3 基础权重不存在：{base_weight}")
+
+    command_record = {
+        "module": name,
+        "mode": "subprocess",
+        "argv": command,
+        "cwd": str(runner_path.parent) if runner_path is not None else None,
+        "install_root": str(install_root) if install_root is not None else None,
+        "install_root_read_only": bool(section.get("install_root_read_only", True)),
+        "torch_home": str(torch_home) if torch_home is not None else None,
+        "base_weight": str(base_weight) if base_weight is not None else None,
+        "output": str(output_path),
+        "service_fallback_reason": service_error,
+    }
+    stdout = ""
+    stderr = ""
+    returncode: int | None = None
+    timed_out = False
+    execution_mode = "subprocess"
+    if not errors and runner_path is not None:
+        try:
+            subprocess_timeout = _remaining_timeout(deadline, module=name)
+            command_record["attempt_timeout_seconds"] = subprocess_timeout
+            completed = _run_subprocess(
+                command,
+                cwd=runner_path.parent,
+                timeout=subprocess_timeout,
+                hide_console=bool(section.get("hide_console", False)),
+                executable_worker=executable_worker,
+                extra_environment=(
+                    {"TORCH_HOME": str(torch_home), "PYTHONDONTWRITEBYTECODE": "1"}
+                    if torch_home is not None
+                    else {"PYTHONDONTWRITEBYTECODE": "1"}
+                ),
+            )
+            returncode = completed.returncode
+            stdout = completed.stdout
+            stderr = completed.stderr
+        except subprocess.TimeoutExpired as exc:
+            timed_out = True
+            stdout = exc.stdout or ""
+            stderr = exc.stderr or ""
+            errors.append(f"deepfake 超时（模块总预算 {timeout:g} 秒）")
+        except TimeoutError as exc:
+            timed_out = True
+            execution_mode = "deadline_exhausted"
+            command_record["attempted"] = False
+            errors.append(f"deepfake 超时（模块总预算 {timeout:g} 秒）：{exc}")
+        except OSError as exc:
+            errors.append(f"无法启动 deepfake 子进程：{exc}")
+            stderr = (stderr + "\n" + str(exc)).strip() + "\n"
+    elif errors:
+        stderr = "\n".join(errors) + "\n"
+
+    payload: Any = None
+    if output_path.is_file():
+        try:
+            payload = json.loads(output_path.read_text(encoding="utf-8"))
+            valid, validation_error = _validate_deepfake_result(payload)
+            if not valid:
+                errors.append(validation_error)
+        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
+            errors.append(f"无法解析 deepfake 结果 JSON：{exc}")
+    else:
+        errors.append("deepfake 未生成结果文件")
+    if returncode not in (0, None):
+        errors.append(f"deepfake 子进程退出码为 {returncode}")
+    if isinstance(payload, dict) and payload.get("status") == "error":
+        errors.append(str(payload.get("error_message") or "deepfake 返回 error"))
+    if errors:
+        if output_path.is_file():
+            try:
+                shutil.copy2(output_path, logs_dir / "deepfake_raw_result.json")
+            except OSError:
+                pass
+        if service_error:
+            errors.insert(0, f"持久服务回退原因：{service_error}")
+        message = "; ".join(dict.fromkeys(item.strip() for item in errors if item.strip()))
+        payload = _standard_deepfake_result(query_path, time.perf_counter() - started, message)
+    return _finish_module(
+        name=name,
+        payload=payload,
+        output_path=output_path,
+        logs_dir=logs_dir,
+        started_at=started_at,
+        started=started,
+        returncode=returncode,
+        timed_out=timed_out,
+        stdout=stdout,
+        stderr=stderr,
+        command_record=command_record,
+        execution_mode=execution_mode,
+        fallback_reason=service_error,
+    )
+
+
+def _run_search_module(
+    *,
+    name: str,
+    section: dict[str, Any],
+    config_path: Path,
+    query_path: Path,
+    library_override: str | Path | None,
+    output_path: Path,
+    logs_dir: Path,
+    identity_cache: Path,
+    progress: _ProgressReporter,
+    services: Any = None,
+) -> tuple[dict[str, Any], dict[str, Any]]:
+    started_at = _now_iso()
+    started = time.perf_counter()
+    enabled = section.get("enabled", True)
+    if not isinstance(enabled, bool):
+        message = f"{name} 配置无效：enabled 必须为布尔值"
+        payload = _standard_search_result(query_path, 0.0, message)
+        return _finish_module(
+            name=name,
+            payload=payload,
+            output_path=output_path,
+            logs_dir=logs_dir,
+            started_at=started_at,
+            started=started,
+            returncode=None,
+            timed_out=False,
+            stderr=message + "\n",
+            execution_mode="configuration",
+        )
+    if not enabled:
+        message = f"{name} 模块已由配置关闭，无法执行用户选择的模块"
+        payload = _standard_search_result(query_path, 0.0, message)
+        return _finish_module(
+            name=name,
+            payload=payload,
+            output_path=output_path,
+            logs_dir=logs_dir,
+            started_at=started_at,
+            started=started,
+            returncode=None,
+            timed_out=False,
+            stderr=message + "\n",
+            command_record={
+                "module": name,
+                "mode": "disabled",
+                "argv": [],
+                "reason": message,
+                "output": str(output_path),
+            },
+            execution_mode="disabled",
+        )
+    try:
+        library_path = resolve_path(library_override or section["library"], config_path)
+        timeout = float(section.get("timeout_seconds", 600))
+        top_k = int(section.get("top_k", 5))
+    except (KeyError, TypeError, ValueError) as exc:
+        message = f"{name} 配置无效：{exc}"
+        payload = _standard_search_result(query_path, 0.0, message)
+        return _finish_module(
+            name=name,
+            payload=payload,
+            output_path=output_path,
+            logs_dir=logs_dir,
+            started_at=started_at,
+            started=started,
+            returncode=None,
+            timed_out=False,
+            stderr=message + "\n",
+            execution_mode="configuration",
+        )
+
+    timeout_valid = math.isfinite(timeout) and timeout > 0
+    deadline = time.monotonic() + timeout if timeout_valid else None
+    preflight_errors: list[str] = []
+    if not library_path.is_dir():
+        preflight_errors.append(f"图库不存在或不是目录：{library_path}")
+    elif not _library_has_images(library_path):
+        preflight_errors.append(f"图库为空或没有受支持图片：{library_path}")
+    if top_k < 1:
+        preflight_errors.append(f"top_k 必须大于 0：{top_k}")
+    if not timeout_valid:
+        preflight_errors.append(
+            f"timeout_seconds 必须是大于 0 的有限数值：{timeout}"
+        )
+    if preflight_errors:
+        message = "; ".join(preflight_errors)
+        payload = _standard_search_result(query_path, time.perf_counter() - started, message)
+        return _finish_module(
+            name=name,
+            payload=payload,
+            output_path=output_path,
+            logs_dir=logs_dir,
+            started_at=started_at,
+            started=started,
+            returncode=None,
+            timed_out=False,
+            stderr=message + "\n",
+            command_record={
+                "module": name,
+                "mode": "preflight",
+                "library": str(library_path),
+                "output": str(output_path),
+            },
+            execution_mode="preflight",
+        )
+
+    assert deadline is not None
+    if name == "identity":
+        progress.emit("detecting_face", "正在检测人脸……", name)
+        progress.emit("extracting_identity", "正在提取身份特征……", name)
+        progress.emit("searching_identity", "正在搜索身份数据库……", name)
+    else:
+        progress.emit("extracting_source", "正在提取底图特征……", name)
+        progress.emit("searching_source", "正在搜索底图库……", name)
+
+    service_error: str | None = None
+    method = _service_method(services, "run_identity") if name == "identity" else None
+    if method is not None:
+        service_output_path = output_path.with_name(".identity_service_result.json")
+        try:
+            service_timeout = _remaining_timeout(deadline, module=name)
+            payload = _invoke_service(
+                method,
+                kwargs={
+                    "query_path": query_path,
+                    "output_path": service_output_path,
+                    "library_path": library_path,
+                    "top_k": top_k,
+                    "timeout": service_timeout,
+                },
+                timeout=service_timeout,
+            )
+            valid, validation_error = _validate_search_result(payload)
+            if not valid:
+                raise ValueError(validation_error)
+            return _finish_module(
+                name=name,
+                payload=payload,
+                output_path=output_path,
+                logs_dir=logs_dir,
+                started_at=started_at,
+                started=started,
+                returncode=None,
+                timed_out=False,
+                command_record={
+                    "module": name,
+                    "mode": "service",
+                    "method": "run_identity",
+                    "library": str(library_path),
+                    "top_k": top_k,
+                    "timeout_seconds": timeout,
+                    "attempt_timeout_seconds": service_timeout,
+                    "service_output": str(service_output_path),
+                    "output": str(output_path),
+                },
+                execution_mode="service",
+            )
+        except Exception as exc:
+            service_error = f"{type(exc).__name__}: {exc}"
+            progress.emit("identity_service_fallback", "身份常驻服务不可用，切换隔离子进程……", name)
+
+    errors: list[str] = []
+    executable_worker = False
+    executable_path: Path | None = None
+    python_path: Path | None = None
+    runner_path: Path | None = None
+    command: list[str] = []
+    module_cwd = config_path.parent
+    try:
+        executable_value = section.get("executable")
+        if executable_value:
+            executable_path = resolve_path(executable_value, config_path)
+            command_prefix = [str(executable_path)]
+            module_cwd = executable_path.parent
+            executable_worker = True
+        else:
+            python_path = resolve_path(section["python"], config_path)
+            runner_path = resolve_path(section["runner"], config_path)
+            python_args = section.get("python_args", ["-B"])
+            if not isinstance(python_args, list) or not all(isinstance(item, str) for item in python_args):
+                raise ValueError("python_args 必须为字符串数组")
+            command_prefix = [str(python_path), *python_args, str(runner_path)]
+            module_cwd = runner_path.parent
+        command = [
+            *command_prefix,
+            "--query",
+            str(query_path),
+            "--library",
+            str(library_path),
+            "--top-k",
+            str(top_k),
+            "--output",
+            str(output_path),
+        ]
+    except (KeyError, TypeError, ValueError) as exc:
+        errors.append(f"{name} 配置无效：{exc}")
+
+    if executable_worker:
+        if executable_path is None or not executable_path.is_file():
+            errors.append(f"模块可执行文件不存在：{executable_path}")
+    else:
+        if python_path is None or not python_path.is_file():
+            errors.append(f"Python 不存在：{python_path}")
+        if runner_path is None or not runner_path.is_file():
+            errors.append(f"runner 不存在：{runner_path}")
+    if name == "identity":
+        try:
+            identity_cache.mkdir(parents=True, exist_ok=True)
+            command.extend(["--ctx", str(section.get("ctx", -1))])
+            command.extend(["--cache-dir", str(identity_cache)])
+            if section.get("model_root"):
+                command.extend(["--model-root", str(resolve_path(section["model_root"], config_path))])
+            if section.get("offline", True):
+                command.append("--offline")
+        except OSError as exc:
+            errors.append(f"无法创建身份缓存目录：{exc}")
+
+    command_record = {
+        "module": name,
+        "mode": "subprocess",
+        "argv": command,
+        "cwd": str(module_cwd),
+        "library": str(library_path),
+        "output": str(output_path),
+        "service_fallback_reason": service_error,
+    }
+    returncode: int | None = None
+    timed_out = False
+    execution_mode = "subprocess"
+    stdout = ""
+    stderr = ""
+    if not errors:
+        try:
+            subprocess_timeout = _remaining_timeout(deadline, module=name)
+            command_record["attempt_timeout_seconds"] = subprocess_timeout
+            completed = _run_subprocess(
+                command,
+                cwd=module_cwd,
+                timeout=subprocess_timeout,
+                hide_console=bool(section.get("hide_console", False)),
+                executable_worker=executable_worker,
+            )
+            returncode = completed.returncode
+            stdout = completed.stdout
+            stderr = completed.stderr
+        except subprocess.TimeoutExpired as exc:
+            timed_out = True
+            stdout = exc.stdout or ""
+            stderr = exc.stderr or ""
+            errors.append(f"{name} 超时（模块总预算 {timeout:g} 秒）")
+        except TimeoutError as exc:
+            timed_out = True
+            execution_mode = "deadline_exhausted"
+            command_record["attempted"] = False
+            errors.append(f"{name} 超时（模块总预算 {timeout:g} 秒）：{exc}")
+        except OSError as exc:
+            errors.append(f"无法启动 {name} 子进程：{exc}")
+            stderr = (stderr + "\n" + str(exc)).strip() + "\n"
+    else:
+        stderr = "\n".join(errors) + "\n"
+
+    payload: Any = None
+    if output_path.is_file():
+        try:
+            payload = json.loads(output_path.read_text(encoding="utf-8"))
+            valid, validation_error = _validate_search_result(payload)
+            if not valid:
+                errors.append(validation_error)
+        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
+            errors.append(f"无法解析 {name} 结果 JSON：{exc}")
+    else:
+        errors.append(f"{name} 未生成结果文件")
+    if returncode not in (0, None):
+        errors.append(f"{name} 子进程退出码为 {returncode}")
+    if isinstance(payload, dict) and payload.get("status") == "error":
+        errors.append(str(payload.get("error_message") or f"{name} 返回 error"))
+    if errors:
+        if output_path.is_file():
+            try:
+                shutil.copy2(output_path, logs_dir / f"{name}_raw_result.json")
+            except OSError:
+                pass
+        if service_error:
+            errors.insert(0, f"持久服务回退原因：{service_error}")
+        message = "; ".join(dict.fromkeys(item.strip() for item in errors if item.strip()))
+        payload = _standard_search_result(query_path, time.perf_counter() - started, message)
+    return _finish_module(
+        name=name,
+        payload=payload,
+        output_path=output_path,
+        logs_dir=logs_dir,
+        started_at=started_at,
+        started=started,
+        returncode=returncode,
+        timed_out=timed_out,
+        stdout=stdout,
+        stderr=stderr,
+        command_record=command_record,
+        execution_mode=execution_mode,
+        fallback_reason=service_error,
+    )
+
+
+def _unexpected_module_failure(
+    *,
+    name: str,
+    query_path: Path,
+    output_path: Path,
+    logs_dir: Path,
+    exc: Exception,
+) -> tuple[dict[str, Any], dict[str, Any]]:
+    started_at = _now_iso()
+    started = time.perf_counter()
+    message = f"{name} 分支发生受控异常：{type(exc).__name__}: {exc}"
+    payload = (
+        _standard_deepfake_result(query_path, 0.0, message)
+        if name == "deepfake"
+        else _standard_search_result(query_path, 0.0, message)
+    )
+    return _finish_module(
+        name=name,
+        payload=payload,
+        output_path=output_path,
+        logs_dir=logs_dir,
+        started_at=started_at,
+        started=started,
+        returncode=None,
+        timed_out=False,
+        stderr=message + "\n",
+        command_record={"module": name, "mode": "orchestrator_exception", "argv": []},
+        execution_mode="orchestrator_exception",
+    )
+
+
+def _normalize_selected_modules(selected_modules: Collection[str] | None) -> frozenset[str]:
+    if selected_modules is None:
+        return frozenset(MODULE_ORDER)
+    if isinstance(selected_modules, str):
+        values = {selected_modules}
+    else:
+        try:
+            values = {str(value) for value in selected_modules}
+        except TypeError as exc:
+            raise OrchestrationError("selected_modules 必须是模块名集合。") from exc
+    unknown = sorted(values - set(MODULE_ORDER))
+    if unknown:
+        raise OrchestrationError(
+            "selected_modules 包含未知模块："
+            + ", ".join(unknown)
+            + "；可选值为 deepfake/identity/source。"
+        )
+    if not values:
+        raise OrchestrationError("selected_modules 至少必须包含一个模块。")
+    return frozenset(values)
 
 
 def run_task(
@@ -352,97 +1124,150 @@
     *,
     identity_library: str | Path | None = None,
     source_library: str | Path | None = None,
+    selected_modules: Collection[str] | None = None,
+    services: RuntimeServices | Mapping[str, Callable[..., Any]] | Any | None = None,
     config_path: str | Path | None = None,
     progress_callback: ProgressCallback | None = None,
 ) -> dict[str, Any]:
-    progress = progress_callback or (lambda _message: None)
+    selected = _normalize_selected_modules(selected_modules)
     config, resolved_config = load_config(config_path)
     query_source = Path(query).expanduser().resolve(strict=False)
     if not query_source.is_file():
         raise OrchestrationError(f"查询图片不存在或不是文件：{query_source}")
 
-    tasks_root = resolve_path(config["runtime"]["tasks_root"], resolved_config)
-    identity_cache = resolve_path(config["runtime"]["identity_cache"], resolved_config)
+    try:
+        tasks_root = resolve_path(config["runtime"]["tasks_root"], resolved_config)
+        identity_cache = resolve_path(config["runtime"]["identity_cache"], resolved_config)
+    except (KeyError, TypeError, ValueError) as exc:
+        raise OrchestrationError(f"runtime 配置无效：{exc}") from exc
     task_id, task_dir = _new_task_dir(tasks_root)
     logs_dir = task_dir / "logs"
     logs_dir.mkdir(parents=True, exist_ok=True)
+    (task_dir / "visualizations").mkdir(parents=True, exist_ok=True)
     query_path = task_dir / "query.jpg"
     shutil.copy2(query_source, query_path)
 
     started_at = _now_iso()
     started = time.perf_counter()
-    progress("正在创建任务……")
+    progress = _ProgressReporter(progress_callback, logs_dir / "progress.json")
+    progress.emit("task_created", "正在创建任务……")
     _write_json(
         task_dir / "task_status.json",
         {
+            "schema_version": "2.0",
             "task_id": task_id,
             "status": "running",
             "started_at": started_at,
             "query_source": str(query_source),
             "query_image": str(query_path),
+            "selected_modules": [name for name in MODULE_ORDER if name in selected],
         },
     )
 
-    identity_result, identity_meta = _run_module(
-        name="identity",
-        section=config["identity"],
-        config_path=resolved_config,
-        query_path=query_path,
-        library_override=identity_library,
-        output_path=task_dir / "identity_result.json",
-        logs_dir=logs_dir,
-        identity_cache=identity_cache,
-        progress=progress,
+    output_paths = {
+        "deepfake": task_dir / "detection_result.json",
+        "identity": task_dir / "identity_result.json",
+        "source": task_dir / "source_result.json",
+    }
+    results: dict[str, dict[str, Any]] = {}
+    metadata: dict[str, dict[str, Any]] = {}
+    for name in MODULE_ORDER:
+        if name not in selected:
+            results[name], metadata[name] = _skip_module(
+                name=name,
+                query_path=query_path,
+                output_path=output_paths[name],
+                logs_dir=logs_dir,
+                reason="用户未选择此模块",
+            )
+            progress.emit(f"{name}_skipped", f"{name} 模块已跳过。", name)
+            continue
+        try:
+            if name == "deepfake":
+                results[name], metadata[name] = _run_deepfake_module(
+                    section=config[name],
+                    config_path=resolved_config,
+                    query_path=query_path,
+                    output_path=output_paths[name],
+                    logs_dir=logs_dir,
+                    progress=progress,
+                    services=services,
+                )
+            else:
+                results[name], metadata[name] = _run_search_module(
+                    name=name,
+                    section=config[name],
+                    config_path=resolved_config,
+                    query_path=query_path,
+                    library_override=(identity_library if name == "identity" else source_library),
+                    output_path=output_paths[name],
+                    logs_dir=logs_dir,
+                    identity_cache=identity_cache,
+                    progress=progress,
+                    services=services,
+                )
+        except Exception as exc:
+            results[name], metadata[name] = _unexpected_module_failure(
+                name=name,
+                query_path=query_path,
+                output_path=output_paths[name],
+                logs_dir=logs_dir,
+                exc=exc,
+            )
+        progress.emit(
+            f"{name}_finished",
+            f"{name} 模块完成：{results[name]['status']}",
+            name,
+        )
+
+    selected_statuses = [results[name]["status"] for name in MODULE_ORDER if name in selected]
+    successes = selected_statuses.count("success")
+    non_successes = sum(status != "success" for status in selected_statuses)
+    overall_status = (
+        "success"
+        if non_successes == 0
+        else "partial_success"
+        if successes > 0
+        else "error"
     )
-    source_result, source_meta = _run_module(
-        name="source",
-        section=config["source"],
-        config_path=resolved_config,
-        query_path=query_path,
-        library_override=source_library,
-        output_path=task_dir / "source_result.json",
-        logs_dir=logs_dir,
-        identity_cache=identity_cache,
-        progress=progress,
-    )
-
-    successful = sum(
-        result.get("status") == "success"
-        for result in (identity_result, source_result)
-    )
-    overall_status = "success" if successful == 2 else "partial_success" if successful == 1 else "error"
     total_runtime = time.perf_counter() - started
     finished_at = _now_iso()
     errors = [
-        f"{module}: {result['error_message']}"
-        for module, result in (
-            ("identity", identity_result),
-            ("source", source_result),
-        )
-        if result.get("status") == "error"
+        f"{name}: {results[name]['error_message']}"
+        for name in MODULE_ORDER
+        if name in selected and results[name].get("status") == "error"
     ]
     unified = {
-        "schema_version": "1.0",
+        "schema_version": "2.0",
         "task_id": task_id,
         "status": overall_status,
         "query_image": "query.jpg",
+        "selected_modules": [name for name in MODULE_ORDER if name in selected],
+        "deepfake_detection": {
+            "selected": "deepfake" in selected,
+            "status": results["deepfake"]["status"],
+            "result_file": "detection_result.json",
+        },
         "identity_search": {
-            "status": identity_result["status"],
+            "selected": "identity" in selected,
+            "status": results["identity"]["status"],
             "result_file": "identity_result.json",
         },
         "source_search": {
-            "status": source_result["status"],
+            "selected": "source" in selected,
+            "status": results["source"]["status"],
             "result_file": "source_result.json",
         },
         "runtime_seconds": {
-            "identity": identity_meta["runtime_seconds"],
-            "source": source_meta["runtime_seconds"],
+            **{name: metadata[name]["runtime_seconds"] for name in MODULE_ORDER},
             "total": total_runtime,
         },
         "errors": errors,
     }
     _write_json(task_dir / "unified_result.json", unified)
+    progress.emit("completed", "分析完成。")
     task_status = {
+        "schema_version": "2.0",
         "task_id": task_id,
         "task_dir": str(task_dir),
         "status": overall_status,
@@ -450,9 +1275,13 @@
         "finished_at": finished_at,
         "query_source": str(query_source),
         "query_image": str(query_path),
-        "identity": identity_meta,
-        "source": source_meta,
+        "selected_modules": [name for name in MODULE_ORDER if name in selected],
+        "deepfake": metadata["deepfake"],
+        "identity": metadata["identity"],
+        "source": metadata["source"],
         "runtime_seconds": total_runtime,
+        "progress_log": str(logs_dir / "progress.json"),
+        "errors": errors,
     }
     _write_json(task_dir / "task_status.json", task_status)
     _write_text(
@@ -461,20 +1290,21 @@
             [
                 f"task_id={task_id}",
                 f"status={overall_status}",
-                f"identity_status={identity_result['status']}",
-                f"source_status={source_result['status']}",
+                "selected_modules=" + ",".join(name for name in MODULE_ORDER if name in selected),
+                *[f"{name}_status={results[name]['status']}" for name in MODULE_ORDER],
                 f"runtime_seconds={total_runtime:.6f}",
             ]
         )
         + "\n",
     )
-    progress("分析完成。")
     return task_status
 
 
 __all__ = [
     "DEFAULT_CONFIG",
+    "MODULE_ORDER",
     "OrchestrationError",
+    "RuntimeServices",
     "load_config",
     "resolve_path",
     "run_task",
```

### `run_app.bat`

```diff
--- base/run_app.bat
+++ target/run_app.bat
@@ -1,13 +1,4 @@
 @echo off
-setlocal
-set "PRODUCT_ROOT=%~dp0"
-set "PYTHON_CMD=python"
-py -3.14 -B -c "import streamlit" >nul 2>&1
-if not errorlevel 1 set "PYTHON_CMD=py -3.14"
-%PYTHON_CMD% -B -c "import streamlit" >nul 2>&1
-if errorlevel 1 (
-  echo Error: no verified Python environment with Streamlit was found.
-  exit /b 2
-)
-%PYTHON_CMD% -B -m streamlit run "%PRODUCT_ROOT%frontend\app.py" --server.address=127.0.0.1 --server.port=8766 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false
+setlocal EnableExtensions
+call "%~dp0start.bat"
 exit /b %ERRORLEVEL%
```

### `tests/README.md`

```diff
--- base/tests/README.md
+++ target/tests/README.md
@@ -1,6 +1,9 @@
 # 快速集成测试
 
-这些测试覆盖 task_id 格式与唯一性、缺失 query 的受控错误，以及 A 失败后 B 仍执行并形成 `partial_success`。
+这些测试覆盖 task_id 格式与唯一性、缺失 query 的受控错误、三模块选择与
+`skipped` 契约、空库快速失败、子进程超时、服务回退、结构化进度、系统状态，
+以及持久工作进程的 JSON 行协议。协议测试会验证同一 PID 跨查询复用、
+`ready`/`health`/`request_id` 响应、第三方 stdout 噪音过滤、超时和进程退出检测。
 
 在项目根目录运行：
 
@@ -8,4 +11,5 @@
 D:\刑侦查竞赛\first_week\_round1_work\backend\envs\A_env\Scripts\python.exe -B -m pytest tests -q
 ```
 
-测试仅使用 `tests/fixtures` 中的两个轻量 runner，不会运行真实模型或 200 张底图库。
+测试仅使用 `tests/fixtures` 中的轻量 runner/mock worker，不会初始化或运行
+Deepfake、InsightFace、ONNX Runtime 等真实模型，也不会扫描 200 张底图库。
```

### `tests/fixtures/README.md`

```diff
--- base/tests/fixtures/README.md
+++ target/tests/fixtures/README.md
@@ -1,8 +1,10 @@
 # 轻量 runner fixtures
 
-这两个脚本只用于验证 orchestrator 的子进程故障隔离：
+这些脚本只用于验证 orchestrator 的子进程故障隔离和服务协议：
 
 - `failing_identity_runner.py` 写入标准错误结果并返回非零退出码；
 - `successful_source_runner.py` 写入一个最小成功结果，并输出执行标记。
+- `slow_search_runner.py` 休眠，用于验证受控 timeout；
+- `mock_persistent_worker.py` 提供带噪声 stdout 的 JSON 行协议假 worker。
 
 它们不导入 A/B 模型、不解码查询图片、不读取真实图库，也不包含或替代任何搜索算法。
```

### `tests/test_orchestrator_contract.py`

```diff
--- base/tests/test_orchestrator_contract.py
+++ target/tests/test_orchestrator_contract.py
@@ -36,6 +36,9 @@
     source_library = tmp_path / "source_library"
     identity_library.mkdir()
     source_library.mkdir()
+    (identity_library / "person_001").mkdir()
+    (identity_library / "person_001" / "reference.jpg").write_bytes(b"fixture")
+    (source_library / "source.jpg").write_bytes(b"fixture")
 
     config = {
         "schema_version": "test",
@@ -43,6 +46,10 @@
             "tasks_root": str(tmp_path / "tasks"),
             "inbox_root": str(tmp_path / "inbox"),
             "identity_cache": str(tmp_path / "cache"),
+        },
+        "deepfake": {
+            "enabled": False,
+            "timeout_seconds": 10,
         },
         "identity": {
             "python": sys.executable,
@@ -122,6 +129,7 @@
     unified = json.loads((task_dir / "unified_result.json").read_text(encoding="utf-8"))
 
     assert status["status"] == "partial_success"
+    assert status["deepfake"]["status"] == "error"
     assert status["identity"]["status"] == "error"
     assert status["source"]["status"] == "success"
     assert identity_result["status"] == "error"
```
