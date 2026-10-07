/**
 * @author wzhikai@linewell.com
 * @since 2020-09-03
 */
;
define(function (require, exports, module) {

	var appAjax = require("app-ajax");

	var artTemplate = require("artTemplate");

	var notify = require("ai/js/utils/notify");

	var utils = require("common/js/utils/utils.js");

	var cookie = require("common/js/utils/cookie");

	var id = utils.getUrlParameter("id");

	require("common/js/utils/echarts.min");

	// 成绩配置文件
	var gradeConfig = require("./grade-configure");
	var gradeConfigSecond = require("./grade-configure-second");
	var gradeConfigList = require("./grade-configure-list");

	var currentIndex = 0;

	/**
	 * 内部方法
	 */
	var privateMethods = {

		/**
		 * 渲染项目
		 */
		renderItem: function () {
			remoteOperation.getItem("21ff82af3f9c4a118f1428abcd98f862", function (ret) {
				$("#categoryBox").html(artTemplate("categoryTemp", {
					list: ret || {}
				}));
				$("#categoryBox li").eq(0).click();
			});
		},

		/**
		 * 比赛介绍 / 规则标题加粗、突出
		 */
		enhanceArticleTitles: function ($root, titles, cls) {
			if (!$root || !$root.length) {
				return;
			}
			$root.find("h1,h2,h3,h4,h5,strong,b,p,span,div").each(function () {
				var $el = $(this);
				if ($el.hasClass(cls)) {
					return;
				}
				var text = $.trim($el.text()).replace(/[：:]\s*$/, "");
				if (titles.indexOf(text) !== -1) {
					$el.addClass(cls);
				}
			});
		},

		/**
		 * 判断是否是pc端
		 */
		isPC: function () {
			var userAgentInfo = navigator.userAgent;
			var Agents = ["Android", "iPhone",
				"SymbianOS", "Windows Phone",
				"iPad", "iPod"];
			var flag = true;
			for (var v = 0; v < Agents.length; v++) {
				if (userAgentInfo.indexOf(Agents[v]) > 0) {
					flag = false;
					break;
				}
			}
			return flag;
		},
	};

	/**
	 * 远程数据操作
	 */
	var remoteOperation = {

		/**
		 * 获取分类
		 */
		getItem: function (id, callback) {
			appAjax.postJson({
				service: "GET_SON_CATEGORY_LIST",
				data: {
					id: id
				},
				success: function (ret) {
					callback && callback(ret);
				}
			});
		},

		getInfo: function () {
			appAjax.postJson({
				service: "GET_COMPETITION_INFO",
				data: {
					id: id
				},
				success: function (data) {
					$("#infoBox").html(artTemplate("infoTemp", {info: data}));

					if (data.competeExt && id != '2026c40dbb2347fba8b3c9a6294efa5b') {

						// 获取大赛介绍文章
						if(data.competeExt.introduceAtricleId) {
							remoteOperation.getIntroduce(data.competeExt.introduceAtricleId, function (data) {
								$("#introduce").html(data.content);
								privateMethods.enhanceArticleTitles($("#introduce"), ["大赛简介", "奖项设置"], "intro-sec-title");
							});
						}

						// 获取比赛规则
						if(data.competeExt.itemDescAtricleId) {
							remoteOperation.getIntroduce(data.competeExt.itemDescAtricleId, function (data) {
								$("#ruleBox").html(data.content);
								privateMethods.enhanceArticleTitles($("#ruleBox"), ["初赛阶段", "决赛阶段"], "rule-stage-title");
							});
						}

						// 获取峰会介绍文章
						if (id == '2022c40dbb2347fba8b3c9a6294efa5b') {
							remoteOperation.getIntroduce('a01f488e2ad44b40bbc09024d9cb8189', function (data) {
								$("#fHintroduce").html(data.content)
							})
						} else if (id == '2200075d26e840b1b9ffd10633d6a9bf') {
							remoteOperation.getIntroduce('80ce0906fc214e51baf322d2ede033a2', function (data) {
								$("#fHintroduce").html(data.content)
							})
						} else if (id == 'a8e0c40dbb2347fba8b3c9a6294efa5b') {
							remoteOperation.getIntroduce('a4883b200d7b4a03b703c3c03cf0c796', function (data) {
								$("#fHintroduce").html(data.content)
							})
						} else if (id != '2026c40dbb2347fba8b3c9a6294efa5b') {
							remoteOperation.getIntroduce('6374c832a7d74466a1dd2161243dbee6', function (data) {
								$("#fHintroduce").html(data.content)
							})
						}
					}

					if (cookie.getCookie('CLICK_SYNY_IN')) {
						$("#tabBar a").eq(2).click();
						cookie.setCookie('CLICK_SYNY_IN', '')
					}

					// 第七届
					if (id == '2026c40dbb2347fba8b3c9a6294efa5b') {
						$(".js-seven").removeClass('hide');
						if (data.competeStatus == 1) {
							$(".js-seven-tab").addClass('hide');
						}
						remoteOperation.getItems(function (data) {
							var list = data || [];
							list.forEach(function (item) {
								if (item.name && item.name.indexOf('家庭陪伴') !== -1) {
									item.name = '公共服务智能交互一体化赛';
								}
							});
							$("#thirdItemsBox").html(artTemplate("itemTemp", {list: list}))
						})
					}
					// 第六届
					else if (id == '2025c40dbb2347fba8b3c9a6294efa5b') {
						$(".js-six").removeClass('hide');
						if (data.competeStatus == 1) {
							$(".js-six-tab").addClass('hide');
						}
						remoteOperation.getItems(function (data) {
							$("#thirdItemsBox").html(artTemplate("itemTemp", {list: data}))
						})
					}
					// 第五届
					else if (id == '2024c40dbb2347fba8b3c9a6294efa5b') {
						$(".js-five").removeClass('hide');
						if (data.competeStatus == 1) {
							$(".js-five-tab").addClass('hide');
						}
						remoteOperation.getItems(function (data) {
							$("#thirdItemsBox").html(artTemplate("itemTemp", {list: data}))
						})
					}
					// 第四届
					else if (id == '2022c40dbb2347fba8b3c9a6294efa5b') {
						$(".js-four").removeClass('hide');
						if (data.competeStatus == 1) {
							$(".js-four-tab").addClass('hide');
						}
						remoteOperation.getItems(function (data) {
							$("#thirdItemsBox").html(artTemplate("itemTemp", {list: data}))
						})
					} else if (id == '0c8ce75d26e840b1b9ffd10633d6a9bf') {
						$(".js-first").removeClass('hide');
						privateMethods.renderItem();
					} else if (id == '0000075d26e840b1b9ffd10633d6a9bf') {
						$(".js-first-half").removeClass('hide');
						remoteOperation.getItems(function (data) {
							$("#itemBox").html(artTemplate("itemTemp", {list: data}))
						})
					} else if (id == '2200075d26e840b1b9ffd10633d6a9bf') {
						$(".js-second").removeClass('hide');
						remoteOperation.getItems(function (data) {
							$("#itemsBox").html(artTemplate("itemTemp", {list: data}))
						})
					} else if (id == 'a8e0c40dbb2347fba8b3c9a6294efa5b') {
						$(".js-third").removeClass('hide');
						remoteOperation.getItems(function (data) {
							$("#thirdItemsBox").html(artTemplate("itemTemp", {list: data}))
						})
					}

					$(".js-category-box").on("click", "li", function () {
						$(".js-category-box").find("li").removeClass("active");
						$(this).addClass("active");
						currentIndex = $(this).attr("data-index");
						$(".js-grade-box").addClass("hide");
						$(".js-grade-box[index=" + currentIndex + "]").removeClass("hide");
					});
				}
			});
		},

		getItems(callback) {
			appAjax.postJson({
				service: "GET_LIST_SIGN_COMPETE",
				data: {
					id: id
				},
				success: function (data) {
					callback && callback(data)

				}
			});
		},

		getIntroduce(id, callback) {
			appAjax.postJson({
				service: "GET_ARTICLE_DETAIL",
				data: {
					id: id
				},
				success: function (data) {
					if (data && data.content) {
						callback && callback(data);
					}
				},
				error: function () {}
			});
		},

		/**
		 * 获取文章列表
		 * @param {Object} postData
		 * @param {Object} callback
		 */
		getArtical: function (postData, callback) {
			appAjax.postJson({
				service: "GET_ARTICLE_LIST",
				data: postData,
				success: function (ret) {
					callback && callback(ret);
				}
			});
		}
	};

	/**
	 * 初始化数据
	 */
	var _initData = function () {
		remoteOperation.getInfo();
	};

	/**
	 * 初始化视图
	 */
	var _initView = function () {

	};

	/**
	 * 初始化事件
	 */
	var _initEvent = function () {

		$("#infoBox").on("click", "#tabBar a", function () {
			var index = $(this).index();
			$(this).addClass('active').siblings().removeClass('active')
			$(".js-detail-box").addClass('hide')
			$("#contentBox_" + index).removeClass('hide');

			if (index == 4) {

				var $gradeNavByCompete = $("#infoBox").find("#gradeNav" + id);
				if($gradeNavByCompete.length > 0) {
					$gradeNavByCompete.find("li:first").click();
				} else {
					$("#infoBox").find(".js-grade-nav[index=" + currentIndex + "]").click();
				}
			}
		});

		$("#infoBox").on("click", ".js-click-item", function () {
			cookie.setCookie('CLICK_SYNY_IN', 1)
		})

		$("#infoBox").on("click", ".js-sign", function () {
			location.href = INNO_CHINA_CONSTANTS.BASE_URL + "/competition/sign-up.html?comId=" + id;
		})

		// 点击项目分类
		$("#infoBox").on("click", "#categoryBox li", function () {
			var cateId = $(this).attr("cateId");
			$(this).addClass("active").siblings().removeClass("active");
			remoteOperation.getArtical({id: cateId}, function (ret) {
				$("#categoryItemBox").html(artTemplate("categoryItemTemp", {
					list: ret || {},
					competeId: id
				}));
			});
		});

		$("#infoBox").on("click", ".js-grade-nav", function () {
			if ($(this).hasClass('disabled')) {
				notify.showTip("尚未开放", "notice");
				return;
			}
			var index = $(this).attr("index");
			if (index == 0) {
				$(this).addClass("active").siblings().removeClass("active");
				$(".js-category-nav, .js-grade-tip, #dataShowBox, #dataShowBoxSecond, #dataShowBoxThird,.js-echart-box, .js-tab-two").addClass("hide");
				$(".js-tab-one").removeClass("hide");
			} else if (index == 1) {
				$(this).addClass("active").siblings().removeClass("active");
				$(".js-category-nav, .js-grade-tip, #dataShowBox, #dataShowBoxSecond, #dataShowBoxThird,.js-echart-box, .js-tab-one").addClass("hide");
				$(".js-tab-two").removeClass("hide");
			} else {
				$(this).addClass("active").siblings().removeClass("active");
				$(".js-category-nav, .js-grade-tip").removeClass("hide");
				$(".js-category-box[index=" + index + "]").removeClass("hide").siblings().addClass("hide");
				$(".js-tab-one, .js-tab-two").addClass("hide");

				if (index == 2) {
					$(".js-category-box").find("li[data-index=0]").click();
				} else if (index == 3) {
					$(".js-category-box").find("li[data-index=9]").click();
				} else if (index == 4) {
					$(".js-category-box").find("li[data-index=12]").click();
				} else if (index == 5) {
					$(".js-category-box").find("li[data-index=13]").click();
				} else if (index == 6) {
					$(".js-category-box").find("li[data-index=14]").click();
				} else if (index == 15) {
					$(".js-category-box").find("li[data-index=15]").click();
				} else if (index == 16) {
					$(".js-category-box").find("li[data-index=16]").click();
				} else if (index == 11) {
					$(".js-category-box").find("li[data-index=11]").click();
				} else {
					$(".js-category-box").find("li[data-index=" + index + "]").click();
				}
			}
		});

		var newsChart;
		$("#infoBox").on("click", ".js-category-box li", function () {
			$(".js-category-box").find("li").removeClass("active");
			$(this).addClass("active");
			currentIndex = $(this).attr("data-index");
			$("#dataShowBox").html(artTemplate("dataShowTemp", {data: gradeConfig[currentIndex]})).removeClass("hide");
			$("#dataShowBoxSecond").html(artTemplate("dataShowSecTemp", {data: gradeConfigSecond[currentIndex]})).removeClass("hide");

			if (gradeConfigList[id] && gradeConfigList[id][currentIndex]) {
				$("#dataShowBox" + id).html(artTemplate("dataShowSecTemp", {data: gradeConfigList[id][currentIndex]})).removeClass("hide");
			}

			if(currentIndex >= 20) {
				return;
			}

			$(".js-echart-box").removeClass("hide");

			var arr = [];
			for (var i = 0; i < gradeConfig[currentIndex][0].list.length; i++) {
				var obj = [gradeConfig[currentIndex][2].list[i].replace(/%/, ""), gradeConfig[currentIndex][1].list[i].replace(/%/, ""), ""];
				arr.push(obj)
			}

			var yMax;
			if (currentIndex == 5 || currentIndex == 6) {
				yMax = 1.2;
			} else {
				yMax = 100;
			}
			newsChart = echarts.init(document.getElementById('echartBox'));
			newsChart.setOption({
				title: {
					text: $(".js-category-box .active").text(),
					left: 'center'
				},
				graphic: [
					{
						type: 'text',
						z: 100,
						left: '3%',
						top: '10%',
						style: {
							fill: '#999',
							text: [
								gradeConfig[currentIndex][1].name
							].join('\n'),
							font: '14px Microsoft YaHei',
						}
					},
					{
						type: 'text',
						z: 100,
						bottom: '0',
						left: 'center',
						style: {
							fill: '#999',
							text: [
								gradeConfig[currentIndex][2].name
							].join('\n'),
							font: '14px Microsoft YaHei',
						}
					},
				],
				grid: {
					left: '3%',
					right: '4%',
					bottom: '10%',
					containLabel: true
				},
				xAxis: {
					splitLine: {
						lineStyle: {
							type: 'dashed'
						}
					},
					min: 0,
					max: 100
				},
				yAxis: {
					splitLine: {
						lineStyle: {
							type: 'dashed'
						}
					},
					scale: true,
					min: 0,
					max: yMax
				},
				series: [{
					data: arr,
					type: 'scatter',
					symbolSize: 30,
					label: {
						emphasis: {
							show: true,
							formatter: function (param) {
								return param.data[2];
							},
							position: 'top'
						}
					},
					itemStyle: {
						normal: {
							shadowBlur: 10,
							shadowColor: 'rgba(120, 36, 50, 0.5)',
							shadowOffsetY: 5,
							color: new echarts.graphic.RadialGradient(0.4, 0.3, 1, [{
								offset: 0,
								color: 'rgb(251, 118, 123)'
							}, {
								offset: 1,
								color: 'rgb(204, 46, 72)'
							}])
						}
					}
				}]
			});
		});

		$("#infoBox").on("click", ".js-change-size", function () {
			var size = $(this).attr("size");
			newsChart.setOption({
				series: [{
					symbolSize: size
				}]
			});
		});

		$("#infoBox").on("click", ".js-check-pic", function () {
			var url = $(this).attr('url')
			$("#picBox").find('img').attr('src', url)
			$("#picBox").removeClass("hide");
			return false;
		});

		$("body").on("click", function () {
			$("#picBox").addClass("hide");
		});

		// 判断是否是浏览器
		if (privateMethods.isPC()) {
			$("#infoBox").on("click", ".js-play", function () {
				var $this = $(this);
				$this.addClass("hide");
				$this.prev().addClass("hide");
				$this.next().removeClass("hide").trigger("play");
			});
		} else {
			$("#infoBox").on("click", ".js-play", function () {
				var $this = $(this);
				var url = $this.next("video").find("source").attr("src");
				var temp = '<source src="' + url + '" type="video/mp4"></source>';
				$("#h5Video").html(temp);
				$("#h5VideoBox").removeClass("hide");
				$("#h5Video").load();
				$("#h5Video").trigger("play");
			});

			$("#h5VideoBox video").on("click", function (e) {
				e.preventDefault();
			});

			$("#h5VideoBox").not(".js-video-box").on("click", function () {
				$("#h5Video").trigger("pause");
				$("#h5VideoBox").addClass("hide");
			});
		}
	};

	var _init = function () {

		// 初始化视图
		_initView();

		// 初始化数据
		_initData();

		// 初始化事件
		_initEvent();
	};

	$(function () {
		_init();
	});
});
