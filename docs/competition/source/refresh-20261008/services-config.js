
/**
 * 服务的配置
 * @author hlijing@linewell.com
 *
 * @since 2018-05-01
 */
;define("services-config", function(require, exports, module){

	// 通用接口配置
	var commonServiceConfig = {

		// 获取上传参数
		GET_UPLOAD_TOKEN : "/citizencloud/common/getTokenForPC",

		GET_WECHAT_JS_SIGN : "/wx/getWechatJsSign",

		// 获取地区
		GET_AREAR : "/general/config/area/getSonList",

		// 获取站点下的服务
		GET_LIST_SERVICE_BY_SITE : "/citizencloud/service/listServiceBySite",

		// 获取滑动验证码背景图
		GET_IMAGE_VERIFY_INFO : "/citizencloud/compete/account/get-image-verify-info",

		// 验证滑动验证码
		GET_IMAGE_VERIFY_RESULT : "/citizencloud/compete/account/get-image-verify-result",

		LOGIN_OUT_2020_V1: '/citizencloud/compete/account/logout',
	};

	// 首页接口配置
	var indexServiceConfig = {
		
		// 获取比赛列表
		GET_COMPETITION: '/citizencloud/compete/list-issue',
		
		// 获取文章子分类
		GET_SON_CATEGORY_LIST : "/general/config/article/category/getSonCategoryList",

		// 获取分类下文章
		GET_ARTICLE_LIST : "/general/config/article/getArticleListByCategoryId",

		// 获取首页动态文章
		GET_ALL_ARTICLE_LIST : "/general/config/article/getAllArticleListByCategoryId",

		// 获取站点下的服务
		GET_LIST_SERVICE_BY_SITE : "/citizencloud/service/listServiceBySite",

		// 获取项目列表
		GET_LIST_SIGN_COMPETE : "/citizencloud/compete/list-sign-compete-item",

		// 获取活动资质
		GET_LIST_SIGN_APTITUDE : "/citizencloud/compete/list-sign-compete-aptitude",

		// 报名
		COMPETE_SIGN_SAVE : "/citizencloud/compete/sign/save",
		COMPETE_SIGN_SAVE2 : "/citizencloud/compete/sign-flow/save",

		// 编辑报名
		COMPETE_SIGN_UPDATE : "/citizencloud/compete/sign/update",
		COMPETE_SIGN_UPDATE2 : "/citizencloud/compete/sign-flow/update",

		// 创新大赛保存草稿
		COMPETE_SELECTION_SIGN_SAVE_DRAFT : "/citizencloud/compete/selection-sign/save-draft",

		// 创新大赛报名
		COMPETE_SELECTION_SIGN_SAVE : "/citizencloud/compete/selection-sign/save",

		// 创新大赛编辑报名
		COMPETE_SELECTION_SIGN_UPDATE : "/citizencloud/compete/selection-sign/update",

		// 报名详情
		COMPETE_SIGN_DETAIL : "/citizencloud/compete/sign/get",

		// 创新之星报名详情
		COMPETE_SELECTION_SIGN_DETAIL : "/citizencloud/compete/selection-sign/get",

		// 创新之星报名导出
		COMPETE_SELECTION_SIGN_EXPORT : "/citizencloud/compete/selection-sign/export-pdf",

		// 获取单条成绩
		COMPETE_ITEM_SCORE_BY_ITEM : "/citizencloud/compete/item-score/getCompeteItemScoreByItem",

		// 提交
		COMPETE_COMFIRM_GRADE : "/citizencloud/compete/item-score/chooseOptimalResults",

		// 补充材料
		COMPETE_COMFIRM_ENCLOSURE : "/citizencloud/compete/sign/submit-supplement-enclosure",

		// 获取活动概况
		GET_PANDECT : "/citizencloud/compete/get",

		// 获取文章详情
		GET_ARTICLE_DETAIL : "/general/config/article/get",

		// 注册
		COMPETE_ACCOUNT_LOGIN : "/citizencloud/compete/account/login-for-register",

		// 获取成员信息
		GET_MEMBER_DETAIL: "/citizencloud/compete/account/getDetail",

		// 保存定向邀请赛
		SAVE_DIRECTION_SIGN_INFO: "/citizencloud/compete/sign/save-invite",
		SAVE_DIRECTION_SIGN_INFO2: "/citizencloud/compete/sign-flow/save-invite",

		// 获取竞赛信息
		GET_INVITE_INFO: "/citizencloud/compete/get-invite",

		GET_RECORD_LIST: "/citizencloud/compete/sign/sign-item-flow-record",
		
		// 获取成绩
		GET_GRADE: "/citizencloud/compete/query-achievement",
		
		// 获取竞赛列表
		GET_COMPETITION_LIST: '/citizencloud/compete/list',
		
		// 获取竞赛详情
		GET_COMPETITION_INFO: '/citizencloud/compete/detail',
		
		// 获取我的参赛列表
		GET_MY_COMPETITION_LIST: "/citizencloud/compete/sign/sign-compete-list"
	};

	// 登录、注册相关接口
	var loginServiceConfig = {

		// 密码登录
		LOGIN : "/citizencloud/compete/account/login",

		// ixm密码登录
		IXM_LOGIN : "/citizencloud/compete/account/accountPwdLogin",

		// 退出登录
		LOGINOUT : "/user/logout",

		// 公共获取验证码
		USER_GET_CODE : "/citizencloud/compete/account/send-verify-code",

		// 是否注册过用户接口
		VALIDATE_PHONE : "/citizencloud/compete/account/account-exist",

		// 注册
		USER_REGISTER : "/citizencloud/compete/account/user-register",

		// 找回密码-发送验证码
		GET_FORGOT_PWD_VERTIFICODE : "/citizencloud/compete/account/send-reset-verify-code",

		// 找回密码
		RESET_PASSWORD : "/citizencloud/compete/account/reset-password",

		// 修改密码
		CHANGE_PASSWORD : "/citizencloud/compete/account/change-password",

		// 获取所属行业列表
		GET_LIST_FORUM_INDUSTRY : "/citizencloud/compete/forum/list-forum-industry",

		// 获取论坛列表
		GET_LIST_FORUM : "/citizencloud/compete/forum/list-forum-info",

		// 论坛报名
		FORUM_SIGN_INFO_SAVE : "/citizencloud/compete/forum/signinfo/save",

		// 论坛报名信息
		FORUM_SIGN_INFO_GET : "/citizencloud/compete/forum/signinfo/info",

		// 酒店列表信息
		FORUM_SIGN_INFO_HOTEL_INFO : "/citizencloud/compete/forum/signinfo/hotelInfo",

		// 论坛密码登录
		FORUM_LOGIN_PW : "/citizencloud/compete/account/login-to-ixm",

		// 获取验证码
		GET_LOGIN_VERTIFICODE : "/citizencloud/compete/account/send-ixm-code",

		// 论坛验证码登录
		FORUM_LOGIN_PIN : "/citizencloud/compete/account/login-to-ixm-by-code",

		// 验证码登录
		ACCOUNT_CODE_LOGIN : "/citizencloud/compete/account/accountCodeLogin",

		// 验证授权码
		SUBMIT_CODE : "/citizencloud/compete/forum/signinfo/vaild-invitation-code",

		//论坛重新报名
		FORUM_SIGN_INFO_RE_SAVE : "/citizencloud/compete/forum/signinfo/re-save",
	};

	// 用户中心相关接口
	var userServiceConfig = {

		// 获取用户信息
		GET_USER_INFO : "/citizencloud/compete/account/get",

		// 更新用户信息
		UPDATE_USER_INFO : "/citizencloud/compete/account/update",

		// 获取我的参赛报名列表
		GET_MY_COMPETE_LIST : "/citizencloud/compete/sign/get-sign-result",

		// 获取我的参赛报名列表-2020
		GET_MY_COMPETE_LIST_TWO : "/citizencloud/compete/sign/get-sign-result-list",

		// 获取创新之星报名列表
		GET_MY_COMPETE_SELECTION_LIST : "/citizencloud/compete/selection-sign/get-sign-result",

		// 获取我的消息列表
		GET_MY_MESSAGE_LIST : "/citizencloud/compete/message/list-my-message",

		// 更新我的消息状态
		UPDATE_MY_MESSAGE : "/citizencloud/compete/message/update-read-status",

		// 获取我的消息未读/已读数量
		GET_MY_MESSAGE_LENGTH : "/citizencloud/compete/message/count-by-params",

		// 获取我的参赛成绩列表
		GET_MY_SCORE_LIST : "/citizencloud/compete/item-score/getAccountScoreListByItem",

		// 获取我的参赛成绩列表-2020
		GET_MY_SCORE_LIST_TWO : "/citizencloud/compete/item-score/getAccountAllScoreListByItem",

		// 保存申诉
		SAVE_APPEAL : "/citizencloud/compete/appeal/save",

        // 用户是否报名
        CHECK_USER_HAS_SIGN : "/citizencloud/compete/sign/get-sign-condition",

        // 发送查看项目账号的验证码
        GET_ACCOUNT_VERTIFY_CODE : "/citizencloud/compete/sign/send-item-account-vertify-code",

        // 获取竞赛账号列表
        GET_SIGN_ACCOUNT_INFO : "/citizencloud/compete/sign/list-sign-account-info",

        // 活动撤回
        // COMPETE_WITHDRAW : "/citizencloud/compete/sign/withdraw",
        COMPETE_WITHDRAW : "/citizencloud/compete/sign-flow/withdraw",

        // 创新之星撤回
        COMPETE_SELECTION_WITHDRAW : "/citizencloud/compete/selection-sign/withdraw",

        // 获取领域列表
        COMPETE_GET_DOMIAIN : "/citizencloud/compete/selection-sign/getDomainConfigs",

		// 提交案例赛材料
		SUBMIT_CASE_MATERIALS : "/citizencloud/compete/sign-flow/submit-case-material",

	};

    // 训练申请数据
    var trainingServiceConfig = {

    	// 训练数据类型列表详情
		GET_TRAINING_DATA_TYPE: "/citizencloud/compete/train/data-type-list",

		// 数据提交
		TRAINING_DATA_APPLY: "/citizencloud/compete/train/save",

		// 我的申请
		GET_MY_TRAINING_LIST: "/citizencloud/compete/train/apply-list",

        // 发送验证码
        SEND_VERIFY_CODE: "/citizencloud/compete/account/send-verify-account-count",

        // 获取详情
        GET_TRAINING_DATA_DETAIL: "/citizencloud/compete/train/apply-info",

		// 验证验证码正确性
		VERIFY_ACCOUNT: "/citizencloud/compete/account/verify-account",
    };

    //模块对外提供的公共方法
    var servicesConfig = $.extend(true, {},
		commonServiceConfig,
		indexServiceConfig,
		userServiceConfig,
		loginServiceConfig,
        trainingServiceConfig
    );

    module.exports = servicesConfig;
});
