/**
 * 用户中心-会员信息-人脸认证
 * @author wchao@linewell.com
 * @since 2018-05-21
 */
;
define(function(require, exports, module) {

	// 异步请求工具类
	var appAjax = require("app-ajax");

	var notify = require("ai/js/utils/notify");

	// 表单渲染控件
	var tableRender = require("ai/js/utils/table-render");

    var utils = require("common/js/utils/utils.js");

    var competeId = utils.getUrlParameter("competeId");

	// 表单对象
	var listTable;

	// 是否登陆标识
	var loginCheck = false;

	/**
	 * 内部方法
	 */
	var privateMethods = {

	};

	/**
	 * 远程数据操作
	 */
	var remoteOperation = {

		getUserInfo : function(){
			appAjax.postJson({
				service : "GET_USER_INFO",
				success : function(data){

					if(data) {
						loginCheck = true;
					}

				},
				error : function() {

				}
			});
		},

		UserhasSign : function(id, callback){
			appAjax.postJson({
				service : "CHECK_USER_HAS_SIGN",
				type : "POST",
				data : {
					id : id
				},
				success : function(data){

					if(!data.hasSign || (data.hasSign && (data.status != 2))){
						notify.showTip("需通过该项目的报名才可下载");
					}
					if(data.hasSign && data.status == 2){
						callback && callback();
					}

				},
				error : function() {
					location.href = "./../login/login.html";
				}
			});
		}

	};

	/**
	 * 初始化数据
	 */
	var _initData = function() {
		remoteOperation.getUserInfo();
	};

	/**
	 * 初始化视图
	 */
	var _initView = function() {

	};

	/**
	 * 初始化事件
	 */
	var _initEvent = function() {

		$(".tabs ul li").eq(0).on("click", function (){
			$(this).addClass("active");
			$(".tabs ul li").eq(1).removeClass("active");
			$(".protocol-detail").removeClass("hide");
			$(".project-list").addClass("hide");
		});

		$(".tabs ul li").eq(1).on("click", function (){
			$(this).addClass("active");
			$(".tabs ul li").eq(0).removeClass("active");
			$(".protocol-detail").addClass("hide");
			$(".project-list").removeClass("hide");
		});

//		$(".tabs ul li").eq(2).on("click", function (){
//			if(loginCheck){
//				location.href = "./../index/sign-up.html";
//			}else{
//				location.href = "./../login/login.html";
//			}
//		})

		$(".js-download").on("click", function (){
			if($(window).width() < 768){
				notify.showTip("请使用PC端进行下载", "notice");
				return;
			}
			var $this = $(this);
//			remoteOperation.UserhasSign($(this).attr("id"), function (){
				location.href = $this.attr("downloadSrc")
//			})
		});

		$(".opera-box a.btn").on("click", function (){
			if(loginCheck){
				location.href= INNO_CHINA_CONSTANTS.BASE_URL + "/competition/sign-up.html?comId="+competeId;
			}else{
				location.href = "./../login/login.html";
			}
		})



	};

	var _init = function() {

		// 初始化视图
		_initView();

		// 初始化数据
		_initData();

		// 初始化事件
		_initEvent();
	};

	$(function() {
		_init();
	});
});
