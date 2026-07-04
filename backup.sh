####################################
# git
####################################
VERSION_NAME="新333"


git status # 查看状态
git add . # 添加所有修改
git commit -m "${VERSION_NAME}" # 提交到本地（替换提交说明）
# git pull origin main # 拉取远程最新代码
git push origin main # 推送到远程\
echo "✅ 成功：代码已提交并同步到远程分支！"

echo "🎉 所有操作执行完成！"


####################################
# 拷贝
####################################
# rm -rf main
# cp -rf _历史代码/version000 main
# cp -rf main "_历史代码/${VERSION_NAME}"
# echo "\n✅ 成功：文件已拷贝！"